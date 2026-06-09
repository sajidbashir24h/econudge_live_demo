from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

DEFAULT_WEIGHTS = {
    "deadstock_savior": 0.10,
    "margin_upsell": 0.10,
    "return_killer": 0.10,
    "logistics_bundler": 0.10,
    "loyalty_multiplier": 0.10,
}
MAX_LIVE_USERS = 50
MAX_CF_CANDIDATES = 220
MAX_COLDSTART_CANDIDATES = 180
TOP_IDS_SCAN = 420


def _clean(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _diversified_user_subset(events_by_user: dict[str, int], user_ids: list[str], max_users: int = MAX_LIVE_USERS) -> list[str]:
    unique_ids = sorted(set([str(uid) for uid in user_ids]))
    if len(unique_ids) <= max_users:
        return unique_ids

    frame = pd.DataFrame({"user_id": unique_ids})
    frame["events"] = frame["user_id"].map(lambda u: int(events_by_user.get(str(u), 0)))
    frame["events"] = pd.to_numeric(frame["events"], errors="coerce").fillna(0).astype(int)

    if frame["events"].nunique() < 3:
        return frame.sample(n=max_users, random_state=42)["user_id"].astype(str).tolist()

    try:
        frame["activity_band"] = pd.qcut(frame["events"].rank(method="first"), q=3, labels=["low", "mid", "high"])
    except Exception:
        return frame.sample(n=max_users, random_state=42)["user_id"].astype(str).tolist()

    selected: list[str] = []
    for band in ["low", "mid", "high"]:
        band_df = frame[frame["activity_band"] == band]
        if band_df.empty:
            continue
        take = min(max(1, max_users // 3), len(band_df))
        selected.extend(band_df.sample(n=take, random_state=42)["user_id"].astype(str).tolist())

    if len(selected) < max_users:
        remaining = frame[~frame["user_id"].isin(selected)]
        if not remaining.empty:
            fill = min(max_users - len(selected), len(remaining))
            selected.extend(remaining.sample(n=fill, random_state=42)["user_id"].astype(str).tolist())

    return selected[:max_users]


def _label_for_user(user_id: str, events_by_user: dict[str, int], unique_by_user: dict[str, int], age_by_user: dict[str, Any], idx: int) -> str:
    events_txt = int(events_by_user.get(str(user_id), 0))
    unique_txt = int(unique_by_user.get(str(user_id), 0))
    age_raw = age_by_user.get(str(user_id), None)
    age_txt = "n/a" if pd.isna(age_raw) else str(int(float(age_raw)))
    return f"User {idx:03d} | purchases {events_txt} | uniq {unique_txt} | age {age_txt}"


def _prepare_frame_for_export(df: pd.DataFrame) -> list[dict[str, Any]]:
    keep_cols = [
        "article_id",
        "prod_name",
        "product_type_name",
        "product_group_name",
        "detail_desc",
        "category",
        "baseline_relevance",
        "popularity",
        "article_recency_days",
        "item_margin_tier_proxy",
        "is_deadstock_proxy",
        "tx_return_risk_mean",
        "tx_logistics_burden_mean",
        "stock_age_days",
        "margin_score",
        "eco_tag",
        "return_probability",
        "distance_km",
        "loyalty_affinity",
    ]
    out = df[[col for col in keep_cols if col in df.columns]].copy()
    return [{key: _clean(value) for key, value in row.items()} for row in out.to_dict(orient="records")]


def _series_or_default(df: pd.DataFrame, column: str, default: float | str = 0.0) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def _build_use_case_candidates(assets, user_id: str, use_case: str) -> dict[str, Any]:
    from src import evaluation, models, utils

    item_meta = assets.item_features_df.copy()
    item_meta["article_id"] = item_meta["article_id"].astype(str)
    item_meta = item_meta.drop_duplicates(subset=["article_id"], keep="first")
    item_meta = evaluation.derive_deadstock_proxy(item_meta)
    item_meta["category"] = item_meta.get("product_group_name", "unknown").fillna("unknown").astype(str)
    item_meta["popularity"] = pd.to_numeric(item_meta.get("item_popularity", 0.0), errors="coerce").fillna(0.0)
    item_meta["stock_age_days"] = pd.to_numeric(item_meta.get("article_recency_days", 0.0), errors="coerce").fillna(0.0).clip(lower=1.0)

    user_profile = utils.build_user_profile(assets, user_id)
    loyalty_affinity = float(user_profile.get("loyalty_proxy", 0.0) or 0.0)

    if use_case == "coldstart_content_fallback":
        query = utils.build_cold_start_query(assets, user_id)
        cold_hits = models.recommend_similar_items_cold_start(assets.cold_start_model, query, top_k=max(MAX_COLDSTART_CANDIDATES, 200))
        ranked = item_meta[item_meta["article_id"].isin([item_id for item_id, _ in cold_hits])].copy()
        ranked = ranked.merge(pd.DataFrame(cold_hits, columns=["article_id", "baseline_relevance"]), on="article_id", how="left")
        ranked["baseline_relevance"] = pd.to_numeric(ranked["baseline_relevance"], errors="coerce").fillna(0.0)
        ranked["margin_score"] = _series_or_default(ranked, "item_margin_tier_proxy", "LOW").astype(str).str.upper().map({"LOW": 0.2, "MID": 0.6, "HIGH": 1.0}).fillna(0.2)
        ranked["eco_tag"] = pd.to_numeric(_series_or_default(ranked, "is_deadstock_proxy", 0.0), errors="coerce").fillna(0.0)
        ranked["return_probability"] = pd.to_numeric(_series_or_default(ranked, "tx_return_risk_mean", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        ranked["distance_km"] = pd.to_numeric(_series_or_default(ranked, "tx_logistics_burden_mean", 0.0), errors="coerce").fillna(0.0) * 1000.0
        ranked["loyalty_affinity"] = loyalty_affinity
        ranked = ranked.sort_values("baseline_relevance", ascending=False).drop_duplicates(subset=["article_id"], keep="first").head(MAX_COLDSTART_CANDIDATES)
        return {"query": query, "candidates": _prepare_frame_for_export(ranked)}

    model = assets.item_item_cf_model if use_case == "item_item_cf" else assets.mf_model
    scores = model.predict_scores(user_id, assets.candidate_items)
    top_ids = models.top_k_from_scores(scores, k=TOP_IDS_SCAN)
    user_pool = assets.strategy_candidate_pool[assets.strategy_candidate_pool["customer_id"].astype(str) == str(user_id)].copy()
    candidate_pool = user_pool[user_pool["article_id"].astype(str).isin(top_ids)].copy()
    if candidate_pool.empty:
        candidate_pool = user_pool.copy()

    candidate_pool["article_id"] = candidate_pool["article_id"].astype(str)
    item_meta_cols = [
        col
        for col in [
            "article_id",
            "prod_name",
            "product_type_name",
            "product_group_name",
            "detail_desc",
            "article_recency_days",
            "item_popularity",
            "tx_return_risk_mean",
            "tx_logistics_burden_mean",
            "item_margin_tier_proxy",
            "is_deadstock_proxy",
        ]
        if col in item_meta.columns
    ]
    candidate_pool = candidate_pool.merge(item_meta[item_meta_cols], on="article_id", how="left", suffixes=("", "_item"))
    for name in ["product_group_name", "item_margin_tier_proxy", "tx_return_risk_mean", "tx_logistics_burden_mean", "article_recency_days", "is_deadstock_proxy"]:
        merged_name = f"{name}_item"
        if merged_name in candidate_pool.columns:
            if name in candidate_pool.columns:
                candidate_pool[name] = candidate_pool[name].where(candidate_pool[name].notna(), candidate_pool[merged_name])
            else:
                candidate_pool[name] = candidate_pool[merged_name]
    if "prod_name_item" in candidate_pool.columns and "prod_name" not in candidate_pool.columns:
        candidate_pool["prod_name"] = candidate_pool["prod_name_item"]
    if "product_type_name_item" in candidate_pool.columns and "product_type_name" not in candidate_pool.columns:
        candidate_pool["product_type_name"] = candidate_pool["product_type_name_item"]
    if "detail_desc_item" in candidate_pool.columns and "detail_desc" not in candidate_pool.columns:
        candidate_pool["detail_desc"] = candidate_pool["detail_desc_item"]
    candidate_pool["baseline_relevance"] = candidate_pool["article_id"].map(scores).fillna(0.0)
    candidate_pool["category"] = _series_or_default(candidate_pool, "product_group_name", "unknown").fillna("unknown").astype(str)
    candidate_pool["popularity"] = pd.to_numeric(_series_or_default(candidate_pool, "item_popularity", 0.0), errors="coerce").fillna(0.0)
    stock_source = candidate_pool["stock_age_days"] if "stock_age_days" in candidate_pool.columns else _series_or_default(candidate_pool, "article_recency_days", 0.0)
    candidate_pool["stock_age_days"] = pd.to_numeric(stock_source, errors="coerce").fillna(0.0).clip(lower=1.0)
    candidate_pool["margin_score"] = pd.to_numeric(_series_or_default(candidate_pool, "margin_score", 0.2), errors="coerce").fillna(0.2)
    candidate_pool["eco_tag"] = pd.to_numeric(_series_or_default(candidate_pool, "eco_tag", 0.0), errors="coerce").fillna(0.0)
    return_source = candidate_pool["return_probability"] if "return_probability" in candidate_pool.columns else _series_or_default(candidate_pool, "tx_return_risk_mean", 0.0)
    candidate_pool["return_probability"] = pd.to_numeric(return_source, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    candidate_pool["distance_km"] = pd.to_numeric(_series_or_default(candidate_pool, "distance_km", 0.0), errors="coerce").fillna(0.0)
    candidate_pool["loyalty_affinity"] = pd.to_numeric(_series_or_default(candidate_pool, "loyalty_affinity", loyalty_affinity), errors="coerce").fillna(loyalty_affinity)
    candidate_pool = candidate_pool.sort_values("baseline_relevance", ascending=False).drop_duplicates(subset=["article_id"], keep="first").head(MAX_CF_CANDIDATES)
    return {"query": None, "candidates": _prepare_frame_for_export(candidate_pool)}


def main() -> None:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    from src import utils

    assets = utils.load_demo_assets()

    train = assets.train_df.copy()
    test = assets.test_df.copy()
    customers = assets.customers_df.copy()
    train["customer_id"] = train["customer_id"].astype(str)
    test["customer_id"] = test["customer_id"].astype(str)
    customers["customer_id"] = customers["customer_id"].astype(str)

    test_users = sorted(test["customer_id"].unique().tolist()) if not test.empty else []
    events_by_user = train.groupby("customer_id").size().rename("events").astype("int64").to_dict() if not train.empty else {}
    unique_by_user = train.groupby("customer_id")["article_id"].nunique().rename("unique_items").astype("int64").to_dict() if not train.empty else {}
    age_by_user = (
        customers.drop_duplicates(subset=["customer_id"], keep="first").set_index("customer_id").get("age", pd.Series(dtype="float64")).to_dict()
    )

    demo_users = _diversified_user_subset(events_by_user, test_users, max_users=MAX_LIVE_USERS)
    if not demo_users:
        demo_users = assets.user_ids[:MAX_LIVE_USERS]

    payload_users: list[dict[str, Any]] = []
    for idx, user_id in enumerate(demo_users, start=1):
        profile = utils.build_user_profile(assets, user_id)
        recent_items = profile.pop("last_items", [])
        payload_users.append(
            {
                "user_id": str(user_id),
                "label": _label_for_user(str(user_id), events_by_user, unique_by_user, age_by_user, idx),
                "profile": {key: _clean(value) for key, value in profile.items()},
                "recent_items": [str(x) for x in recent_items],
                "use_cases": {
                    "item_item_cf": _build_use_case_candidates(assets, str(user_id), "item_item_cf"),
                    "implicit_feedback_confidence": _build_use_case_candidates(assets, str(user_id), "implicit_feedback_confidence"),
                    "coldstart_content_fallback": _build_use_case_candidates(assets, str(user_id), "coldstart_content_fallback"),
                },
            }
        )

    payload = {
        "app_name": "EcoNudge Demo Console",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_size": int(assets.item_features_df["article_id"].astype(str).nunique()),
        "default_weights": DEFAULT_WEIGHTS,
        "strategy_order": list(DEFAULT_WEIGHTS.keys()),
        "demo_users": payload_users,
    }

    assets_dir = ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_path = assets_dir / "demo_bundle.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
