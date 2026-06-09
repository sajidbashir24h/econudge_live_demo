from __future__ import annotations

import html
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "assets" / "demo_bundle.json"

DEFAULT_WEIGHTS = {
    "deadstock_savior": 0.10,
    "margin_upsell": 0.10,
    "return_killer": 0.10,
    "logistics_bundler": 0.10,
    "loyalty_multiplier": 0.10,
}
USE_CASE_LABELS = {
    "item_item_cf": "Item-Item CF baseline",
    "implicit_feedback_confidence": "Implicit feedback confidence",
    "coldstart_content_fallback": "Cold-start content fallback",
}
STRATEGY_TOOLTIPS = {
    "deadstock_savior": (
        "Boosts older inventory to reduce deadstock. Higher weight pushes stale items upward more aggressively. "
        "Typical effect: higher rank shift for older products, higher novelty, and sometimes slightly lower popularity bias if older items are less mainstream. "
        "Trade-off: if pushed too far, relevance can weaken and score gaps versus baseline may become more volatile."
    ),
    "margin_upsell": (
        "Promotes stronger-margin items, blended with the eco tag. Higher weight makes the assortment more business-oriented. "
        "Typical effect: stronger commercial emphasis, but popularity bias can rise if profitable items are also already popular. "
        "Trade-off: novelty and diversity may flatten if the ranking concentrates on proven, high-performing items."
    ),
    "return_killer": (
        "Prioritizes items with lower return-risk signals. Higher weight favors safer purchases and can reduce reverse-logistics pressure. "
        "Typical effect: improved operational stability, more conservative ranking, and sometimes lower popularity bias if risky trend items move down. "
        "Trade-off: if over-weighted, the system may become too cautious and reduce serendipity."
    ),
    "logistics_bundler": (
        "Favors items with lighter logistics-burden proxies, such as more fulfillment-efficient candidates. Higher weight strengthens operational efficiency. "
        "Typical effect: stronger sustainability-through-operations narrative, but popularity bias may increase if efficient items overlap with already common winners. "
        "Trade-off: catalog coverage may not improve much because this strategy optimizes efficiency more than exploration."
    ),
    "loyalty_multiplier": (
        "Prioritizes items aligned with the shopper's loyalty or repeat-interest pattern. Higher weight increases personalization strength. "
        "Typical effect: better perceived relevance and smoother explanations, but popularity bias can increase if the shopper repeatedly prefers mainstream categories. "
        "Trade-off: strong loyalty weighting can reduce diversity and novelty by keeping the ranking close to historical behavior."
    ),
}
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
SLOW_COMPUTE_THRESHOLD_MS = 1200.0
STUCK_COMPUTE_THRESHOLD_MS = 4000.0
STUCK_CONSECUTIVE_RUNS = 2
SAFE_FALLBACK = "This item aligns with your recent preferences and offers a clear practical benefit. We recommend it because it supports a more sustainable and efficient choice based on available evidence."


st.set_page_config(
    page_title="EcoNudge: Sustainable Fashion Recommender for E-Commerce",
    page_icon="EN",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(0, 128, 96, 0.16), transparent 26%),
            radial-gradient(circle at top right, rgba(138, 28, 124, 0.14), transparent 24%),
            linear-gradient(180deg, #07111d 0%, #0b1828 38%, #111827 100%);
        color: #edf2f7;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .eco-hero {
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(10,18,31,0.92), rgba(9,27,23,0.88));
        box-shadow: 0 16px 40px rgba(0,0,0,0.28);
        margin-bottom: 1rem;
    }
    .eco-hero h1, .eco-hero p { margin: 0; }
    .eco-brand {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin-bottom: 0.75rem;
    }
    .eco-logo {
        width: 3.1rem;
        height: 3.1rem;
        border-radius: 18px;
        background: linear-gradient(145deg, #7ee0b8 0%, #008060 52%, #f4c95d 100%);
        box-shadow: 0 12px 30px rgba(0, 128, 96, 0.35);
        display: grid;
        place-items: center;
        color: #07111d;
        font-weight: 900;
        font-size: 1.1rem;
    }
    .eco-brand-copy h1 {
        font-size: 2rem;
        line-height: 1.05;
        letter-spacing: -0.03em;
        color: #ffffff;
    }
    .eco-brand-copy p {
        color: #b9c9da;
        margin-top: 0.25rem;
    }
    .eco-badge {
        display: inline-block;
        padding: 0.26rem 0.62rem;
        border-radius: 999px;
        margin-right: 0.35rem;
        margin-top: 0.4rem;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.08);
        color: #e5eef7;
        font-size: 0.78rem;
    }
    .eco-card {
        background: rgba(9, 15, 25, 0.9);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        height: 100%;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }
    .eco-card-head {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 0.3rem;
    }
    .eco-thumb {
        width: 3.2rem;
        height: 3.2rem;
        border-radius: 15px;
        background: linear-gradient(145deg, rgba(126,224,184,0.95), rgba(0,128,96,0.88));
        color: #07111d;
        display: grid;
        place-items: center;
        font-size: 1rem;
        font-weight: 900;
        flex-shrink: 0;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.2);
    }
    .eco-rank {
        display: block;
        font-weight: 700;
        color: #7ee0b8;
        letter-spacing: 0.02em;
    }
    .eco-title {
        display: block;
        font-size: 1rem;
        font-weight: 700;
        color: #f8fbff;
    }
    .eco-sub {
        color: #87a8c7;
        font-size: 0.84rem;
        margin-bottom: 0.35rem;
    }
    .eco-desc {
        color: #dce6f2;
        font-size: 0.88rem;
        line-height: 1.45;
        margin-bottom: 0.65rem;
        min-height: 3.2em;
    }
    .eco-metadata {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-bottom: 0.55rem;
        color: #d5dde8;
        font-size: 0.78rem;
    }
    .eco-metadata span {
        padding: 0.2rem 0.45rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.07);
    }
    .eco-expl {
        font-size: 0.86rem;
        color: #b9ffdb;
        border-left: 3px solid #15b87b;
        padding-left: 0.6rem;
        min-height: 2.8em;
    }
    .eco-compare-card {
        background: linear-gradient(180deg, rgba(9, 15, 25, 0.95), rgba(9, 15, 25, 0.88));
        border: 1px solid rgba(255,255,255,0.09);
        border-left: 4px solid var(--eco-accent, #008060);
        border-radius: 18px;
        padding: 1rem;
        height: 100%;
    }
    .eco-compare-head {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        align-items: flex-start;
        margin-bottom: 0.8rem;
    }
    .eco-compare-eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.7rem;
        color: #87a8c7;
    }
    .eco-compare-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #f8fbff;
    }
    .eco-compare-pill {
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.12);
        color: #d8e6f4;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .eco-compare-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.6rem;
    }
    .eco-compare-grid div {
        padding: 0.65rem 0.75rem;
        border-radius: 14px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
    }
    .eco-compare-grid span {
        display: block;
        font-size: 0.75rem;
        color: #87a8c7;
        margin-bottom: 0.15rem;
    }
    .eco-compare-grid strong {
        font-size: 1rem;
        color: #f8fbff;
    }
    .help-hero {
        background: linear-gradient(135deg, rgba(0, 128, 96, 0.2), rgba(196, 107, 0, 0.18));
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
    }
    .help-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 0.8rem 0.9rem;
        margin-bottom: 0.6rem;
    }
    .help-card h4 {
        margin: 0 0 0.35rem 0;
        font-size: 0.98rem;
        color: #ffffff;
    }
    .help-card p {
        margin: 0;
        color: #dbe7f4;
        line-height: 1.42;
        font-size: 0.9rem;
    }
    .help-pill {
        display: inline-block;
        padding: 0.12rem 0.44rem;
        border-radius: 999px;
        font-size: 0.74rem;
        margin-right: 0.3rem;
        margin-top: 0.2rem;
        border: 1px solid rgba(255,255,255,0.2);
        background: rgba(255,255,255,0.07);
        color: #f5f9ff;
    }
    .help-section-title {
        margin-top: 0.75rem;
        margin-bottom: 0.3rem;
        color: #ffffff;
        font-size: 1.05rem;
        font-weight: 700;
    }
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4 {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #000000 !important;
    }
    .stTabs [data-baseweb="tab-list"] button {
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: #ff4d4f !important;
    }
    .stTabs [aria-selected="true"] {
        color: #ff4d4f !important;
        border-bottom-color: #ff4d4f !important;
    }
    section[data-testid="stSidebar"] details summary p {
        color: #000000 !important;
        opacity: 0.25;
        transition: opacity 0.15s ease;
    }
    section[data-testid="stSidebar"] details:hover summary p {
        opacity: 1;
    }
    .stAlert [data-testid="stMarkdownContainer"],
    .stAlert p,
    .stAlert div {
        color: #ffffff !important;
    }
    [data-testid="stFormSubmitButton"] button,
    .stButton > button {
        background: linear-gradient(135deg, #0f766e 0%, #059669 100%) !important;
        color: #f8fbff !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 20px rgba(5, 150, 105, 0.25);
    }
    [data-testid="stFormSubmitButton"] button:hover,
    .stButton > button:hover {
        background: linear-gradient(135deg, #0b5e58 0%, #047857 100%) !important;
        color: #f8fbff !important;
        border-color: rgba(255,255,255,0.3) !important;
    }
</style>
"""


@dataclass(frozen=True)
class DemoAssets:
    bundle: dict[str, Any]
    users_by_id: dict[str, dict[str, Any]]
    catalog_size: int
    user_ids: list[str]


@dataclass(frozen=True)
class RecommendationBundle:
    ranked_df: pd.DataFrame
    explanation_text: str
    radar_scores: dict[str, float]
    problem_scores: dict[str, float]
    summary: dict[str, Any]


def _safe_text(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _thumbnail_token(row: pd.Series) -> str:
    label = row.get("prod_name") or row.get("product_type_name") or row.get("category") or row.get("article_id") or "Item"
    parts = [part for part in str(label).split() if part]
    if not parts:
        return "IT"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[1][0]).upper()


def _normalize(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    low = float(numeric.min())
    high = float(numeric.max())
    if np.isclose(low, high):
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    return (numeric - low) / (high - low)


def _minmax(series: pd.Series) -> pd.Series:
    return _normalize(series)


def _series_or_default(df: pd.DataFrame, column: str, default: float | str = 0.0) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def normalize_weights(strategy_weights: dict[str, float]) -> dict[str, float]:
    positive = {k: max(0.0, float(v)) for k, v in strategy_weights.items()}
    total_strat = sum(positive.values())
    if total_strat >= 1.0 and total_strat > 0.0:
        scaled = {k: v / total_strat for k, v in positive.items()}
        return {"baseline_relevance": 0.0, **scaled}
    return {"baseline_relevance": 1.0 - total_strat, **positive}


def _get_secret(name: str) -> str:
    try:
        value = str(st.secrets.get(name, "")).strip()
    except Exception:
        value = ""
    return value or str(os.getenv(name, "")).strip()


@st.cache_resource(show_spinner=False)
def get_gemini_client():
    api_key = _get_secret("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai

        return {"kind": "google_genai", "client": genai.Client(api_key=api_key)}
    except Exception:
        try:
            import google.generativeai as genai_old

            genai_old.configure(api_key=api_key)
            return {"kind": "google_generativeai", "client": genai_old}
        except Exception:
            return None


@st.cache_resource(show_spinner=False)
def load_assets(bundle_mtime: float) -> DemoAssets:
    _ = bundle_mtime
    if not BUNDLE_PATH.exists():
        raise RuntimeError("Missing live/assets/demo_bundle.json. Run live/build_live_bundle.py first.")
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    users = bundle.get("demo_users", [])
    users_by_id = {str(entry["user_id"]): entry for entry in users}
    user_ids = [str(entry["user_id"]) for entry in users]
    return DemoAssets(bundle=bundle, users_by_id=users_by_id, catalog_size=int(bundle.get("catalog_size", 1)), user_ids=user_ids)


def _bundle_mtime() -> float:
    return BUNDLE_PATH.stat().st_mtime if BUNDLE_PATH.exists() else 0.0


def _finite_score_dict(scores: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in scores.items():
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0.0).iloc[0]
        out[str(key)] = float(numeric)
    return out


def build_user_profile(assets: DemoAssets, user_id: str) -> dict[str, Any]:
    user = assets.users_by_id.get(str(user_id), {})
    profile = dict(user.get("profile", {}))
    profile["customer_id"] = str(user_id)
    profile["last_items"] = list(user.get("recent_items", []))
    return profile


def build_cold_start_query(assets: DemoAssets, user_id: str) -> str:
    user = assets.users_by_id.get(str(user_id), {})
    return str(user.get("use_cases", {}).get("coldstart_content_fallback", {}).get("query") or "modern wardrobe essentials")


def _candidate_pool_for_user(assets: DemoAssets, user_id: str, use_case: str) -> pd.DataFrame:
    user = assets.users_by_id.get(str(user_id), {})
    use_case_data = user.get("use_cases", {}).get(use_case, {})
    candidate_df = pd.DataFrame(use_case_data.get("candidates", []))
    if candidate_df.empty:
        return candidate_df
    numeric_cols = [
        "baseline_relevance",
        "popularity",
        "article_recency_days",
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
    for col in numeric_cols:
        if col in candidate_df.columns:
            candidate_df[col] = pd.to_numeric(candidate_df[col], errors="coerce").fillna(0.0)
    candidate_df["article_id"] = candidate_df["article_id"].astype(str)
    if "category" not in candidate_df.columns:
        candidate_df["category"] = _series_or_default(candidate_df, "product_group_name", "unknown").fillna("unknown").astype(str)
    candidate_df["detail_desc"] = _series_or_default(candidate_df, "detail_desc", "").fillna("").astype(str)
    return candidate_df


def _score_deadstock(candidate_df: pd.DataFrame) -> pd.Series:
    return _minmax(candidate_df.get("stock_age_days", pd.Series(0.0, index=candidate_df.index)).astype(float))


def _score_margin(candidate_df: pd.DataFrame) -> pd.Series:
    margin = candidate_df.get("margin_score", pd.Series(0.0, index=candidate_df.index)).astype(float)
    eco = candidate_df.get("eco_tag", pd.Series(0.0, index=candidate_df.index)).astype(float)
    return _minmax(0.7 * margin + 0.3 * eco)


def _score_return(candidate_df: pd.DataFrame) -> pd.Series:
    ret = candidate_df.get("return_probability", pd.Series(1.0, index=candidate_df.index)).astype(float)
    return _minmax(1.0 - ret.clip(0, 1))


def _score_logistics(candidate_df: pd.DataFrame) -> pd.Series:
    dist = candidate_df.get("distance_km", pd.Series(9999.0, index=candidate_df.index)).astype(float)
    raw = np.exp(-dist / 300.0)
    return _minmax(pd.Series(raw, index=candidate_df.index))


def _score_loyalty(candidate_df: pd.DataFrame) -> pd.Series:
    affinity = candidate_df.get("loyalty_affinity", pd.Series(0.0, index=candidate_df.index)).astype(float)
    return _minmax(affinity)


def combine_scores(candidate_df: pd.DataFrame, active_strategies: Sequence[str], strategy_weights: dict[str, float], sustainability_enabled: bool = True) -> pd.DataFrame:
    out = candidate_df.copy()
    if "baseline_relevance" not in out.columns:
        raise ValueError("Missing required column: baseline_relevance")
    if not sustainability_enabled:
        out["final_score"] = pd.to_numeric(out["baseline_relevance"], errors="coerce").fillna(0.0)
        return out.sort_values("final_score", ascending=False)

    active = [s for s in active_strategies if s in DEFAULT_WEIGHTS]
    weights = normalize_weights({k: float(strategy_weights.get(k, 0.0)) for k in active})
    out["final_score"] = weights["baseline_relevance"] * pd.to_numeric(out["baseline_relevance"], errors="coerce").fillna(0.0)
    score_functions = {
        "deadstock_savior": _score_deadstock,
        "margin_upsell": _score_margin,
        "return_killer": _score_return,
        "logistics_bundler": _score_logistics,
        "loyalty_multiplier": _score_loyalty,
    }
    for name in active:
        score_col = f"score_{name}"
        contrib_col = f"contrib_{name}"
        out[score_col] = score_functions[name](out)
        out[contrib_col] = weights[name] * out[score_col]
        out["final_score"] = out["final_score"] + out[contrib_col]
    return out.sort_values("final_score", ascending=False)


def build_explanation(row: pd.Series, active_strategies: Sequence[str], strategy_weights: dict[str, float], sustainable_enabled: bool) -> str:
    title = str(row.get("prod_name") or row.get("product_type_name") or row.get("article_id"))
    category = str(row.get("category") or row.get("product_group_name") or "this category").lower()
    primary_reason, secondary_reason = _explanation_reasons(row, active_strategies, strategy_weights)
    if not sustainable_enabled:
        return f"{title} fits this shopper's current interest in {category}. It remains a strong choice on core relevance signals."
    top_weighted = _weighted_active_strategies(active_strategies, strategy_weights)
    top_strategy = top_weighted[0][0].replace("_", " ").lower() if top_weighted else "baseline relevance"
    return f"{title} fits this shopper's interest in {category}. It stands out because {primary_reason}, with {top_strategy} carrying the strongest strategic emphasis."


def _display_description(row: pd.Series) -> str:
    detail = str(row.get("detail_desc") or "").strip()
    if detail:
        return detail
    prod_name = str(row.get("prod_name") or "").strip()
    product_type = str(row.get("product_type_name") or "").strip()
    category = str(row.get("category") or row.get("product_group_name") or "").strip()
    fallback_parts = [part for part in [prod_name, product_type, category] if part]
    return " | ".join(fallback_parts) if fallback_parts else "No detailed catalog description available."


def _weighted_active_strategies(active_strategies: Sequence[str], strategy_weights: dict[str, float]) -> list[tuple[str, float]]:
    weighted = [(name, float(strategy_weights.get(name, 0.0))) for name in active_strategies if name in DEFAULT_WEIGHTS]
    weighted = [(name, weight) for name, weight in weighted if weight > 0]
    return sorted(weighted, key=lambda item: item[1], reverse=True)


def _strategy_weight_summary(active_strategies: Sequence[str], strategy_weights: dict[str, float]) -> str:
    weighted = _weighted_active_strategies(active_strategies, strategy_weights)
    if not weighted:
        return "baseline relevance remains the main driver"
    labels = [f"{name.replace('_', ' ')} ({weight:.2f})" for name, weight in weighted[:3]]
    return ", ".join(labels)


def _recent_history_text(user_profile: dict[str, Any]) -> str:
    items = [str(x).strip() for x in user_profile.get("last_items", [])[:3] if str(x).strip()]
    return ", ".join(items) if items else "not available"


def _structured_explanation_evidence(row: pd.Series, active_strategies: Sequence[str], strategy_weights: dict[str, float], user_profile: dict[str, Any]) -> dict[str, Any]:
    primary_reason, secondary_reason = _explanation_reasons(row, active_strategies, strategy_weights)
    weighted = _weighted_active_strategies(active_strategies, strategy_weights)
    return {
        "product_name": str(row.get("prod_name") or row.get("product_type_name") or row.get("article_id")),
        "category": str(row.get("category") or row.get("product_group_name") or "unknown"),
        "description": _display_description(row),
        "recent_history": _recent_history_text(user_profile),
        "loyalty_affinity": round(float(user_profile.get("loyalty_proxy", 0.0) or 0.0), 2),
        "top_strategies": [
            {"name": name, "label": name.replace("_", " ").title(), "weight": round(weight, 2)}
            for name, weight in weighted[:3]
        ],
        "primary_reason": primary_reason,
        "secondary_reason": secondary_reason,
    }


def _explanation_reasons(row: pd.Series, active_strategies: Sequence[str], strategy_weights: dict[str, float]) -> tuple[str, str]:
    weighted = _weighted_active_strategies(active_strategies, strategy_weights or {})
    top_order = [name for name, _weight in weighted] or list(active_strategies)

    reason_map = {
        "deadstock_savior": "it helps move older inventory in a more responsible way",
        "margin_upsell": "it balances commercial strength with sustainable assortment goals",
        "return_killer": "it shows lower return-risk signals that support a more confident purchase",
        "logistics_bundler": "it supports a lighter logistics footprint through more efficient fulfillment signals",
        "loyalty_multiplier": "it aligns well with this shopper's repeat-interest patterns",
    }

    selected: list[str] = []
    thresholds = {
        "deadstock_savior": float(row.get("score_deadstock_savior", 0.0)),
        "margin_upsell": float(row.get("score_margin_upsell", 0.0)),
        "return_killer": float(row.get("score_return_killer", 0.0)),
        "logistics_bundler": float(row.get("score_logistics_bundler", 0.0)),
        "loyalty_multiplier": float(row.get("score_loyalty_multiplier", 0.0)),
    }
    for name in top_order:
        if name in reason_map and thresholds.get(name, 0.0) >= 0.45:
            selected.append(reason_map[name])
        if len(selected) == 2:
            break

    if not selected:
        selected.append("it remains strong on baseline relevance")
    if len(selected) == 1:
        selected.append("it still supports a balanced mix of relevance and sustainability")
    return selected[0], selected[1]


def build_gemini_prompt(row: pd.Series, active_strategies: Sequence[str], strategy_weights: dict[str, float], user_profile: dict[str, Any]) -> str:
    weighted_summary = _strategy_weight_summary(active_strategies, strategy_weights)
    evidence = _structured_explanation_evidence(row, active_strategies, strategy_weights, user_profile)
    return (
        "You are writing a premium recommendation explanation for EcoNudge, a sustainable fashion recommender. "
        "Write exactly two sentences in a professional, buyer-facing tone. "
        "Sentence 1 must explain why the item is relevant for the shopper based on category fit, recent interest, or repeat preference patterns. "
        "Sentence 2 must explain how the ranking reflects sustainability or operational priorities from the active strategy controls. "
        "Do not mention raw feature names, contribution columns, internal scores, or backend variable names. "
        "Do not use retailer-only wording like margin score or mid-tier opportunity. "
        "Make the explanation sound natural and persuasive for a shopper considering a purchase.\n\n"
        f"Top active strategy weights: {weighted_summary}\n"
        f"Structured evidence: {evidence}\n"
    )


def generate_explanation(row: pd.Series, active_strategies: Sequence[str], strategy_weights: dict[str, float], user_profile: dict[str, Any]) -> str:
    fallback = build_explanation(row, active_strategies, strategy_weights, True)
    client_info = get_gemini_client()
    if client_info is None:
        return fallback
    model_name = _get_secret("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    prompt = build_gemini_prompt(row, active_strategies, strategy_weights, user_profile)
    try:
        if client_info["kind"] == "google_genai":
            response = client_info["client"].models.generate_content(model=model_name, contents=prompt)
            text = str(getattr(response, "text", "") or "").strip()
        else:
            model = client_info["client"].GenerativeModel(model_name)
            response = model.generate_content(prompt)
            text = str(getattr(response, "text", "") or "").strip()
        return text or fallback
    except Exception:
        return fallback


def recommend_user_bundle(
    assets: DemoAssets,
    user_id: str,
    mode: str,
    use_case: str,
    top_k: int,
    sustainable_enabled: bool,
    active_strategies: Sequence[str],
    strategy_weights: dict[str, float],
    gemma_enabled: bool,
    cold_start_query: str | None = None,
) -> RecommendationBundle:
    candidate_pool = _candidate_pool_for_user(assets, user_id, use_case)
    if candidate_pool.empty:
        raise RuntimeError(f"No candidates available for user_id={user_id} use_case={use_case}")

    if sustainable_enabled and mode == "sustainable" and use_case != "coldstart_content_fallback":
        ranked_df = combine_scores(candidate_pool, active_strategies, strategy_weights, sustainability_enabled=True)
        explanation_text = "Sustainable reranking is active, blending strategy boosts with baseline relevance."
    else:
        ranked_df = candidate_pool.sort_values("baseline_relevance", ascending=False).copy()
        ranked_df["final_score"] = ranked_df["baseline_relevance"]
        explanation_text = "Baseline ranking is active."

    ranked_df = ranked_df.sort_values("final_score", ascending=False).drop_duplicates(subset=["article_id"], keep="first").head(top_k).copy()
    ranked_df["article_id"] = ranked_df["article_id"].astype(str)
    ranked_df["category"] = _series_or_default(ranked_df, "category", "unknown").fillna("unknown").astype(str)
    ranked_df["detail_desc"] = _series_or_default(ranked_df, "detail_desc", "").fillna("").astype(str)
    ranked_df["popularity"] = pd.to_numeric(_series_or_default(ranked_df, "popularity", 0.0), errors="coerce").fillna(0.0)
    return_source = ranked_df["return_probability"] if "return_probability" in ranked_df.columns else _series_or_default(ranked_df, "tx_return_risk_mean", 0.0)
    ranked_df["return_probability"] = pd.to_numeric(return_source, errors="coerce").fillna(0.0)
    ranked_df["distance_km"] = pd.to_numeric(_series_or_default(ranked_df, "distance_km", 0.0), errors="coerce").fillna(0.0)
    ranked_df["loyalty_affinity"] = pd.to_numeric(_series_or_default(ranked_df, "loyalty_affinity", 0.0), errors="coerce").fillna(0.0)
    ranked_df = ranked_df.reset_index(drop=True)
    ranked_df["rank"] = ranked_df.index + 1

    if gemma_enabled and not ranked_df.empty:
        user_profile = build_user_profile(assets, user_id)
        explanation_text = generate_explanation(ranked_df.iloc[0], active_strategies, strategy_weights, user_profile)

    item_count = max(1, int(ranked_df["article_id"].nunique()))
    catalog_count = max(1, assets.catalog_size)
    pop_norm = _normalize(ranked_df["popularity"])
    novelty = float((1.0 - pop_norm).mean())
    diversity = float(ranked_df["category"].nunique() / item_count)
    serendipity = float(((1.0 - pop_norm) * _normalize(ranked_df["final_score"])).mean())
    coverage = float(item_count / catalog_count)
    rank_weights = pd.Series(np.linspace(len(ranked_df), 1, len(ranked_df)), index=ranked_df.index, dtype="float64")
    rank_weights = rank_weights / float(rank_weights.sum())
    popularity_bias = float((pop_norm * rank_weights).sum())
    explainability_score = 1.0 if (gemma_enabled and explanation_text and explanation_text != SAFE_FALLBACK) else 0.0

    problem_scores = {
        "catalog_coverage_gap": float(1.0 - coverage),
        "popularity_bias": popularity_bias,
        "transparency_trust_gap": float(1.0 - explainability_score),
    }
    radar_scores = {
        "Novelty": novelty,
        "Diversity": diversity,
        "Serendipity": serendipity,
        "Explainability": explainability_score,
        "Coverage": coverage,
    }
    summary = {
        "mode": mode,
        "use_case": use_case,
        "gemma_enabled": gemma_enabled,
        "active_strategies": list(active_strategies),
        "strategy_weights": strategy_weights,
    }
    return RecommendationBundle(ranked_df=ranked_df, explanation_text=explanation_text, radar_scores=radar_scores, problem_scores=problem_scores, summary=summary)


def build_radar_figure(scores: dict[str, float], title: str = "Performance Radar") -> go.Figure:
    labels = list(scores.keys())
    values = [float(scores[k]) for k in labels]
    if labels:
        labels.append(labels[0])
        values.append(values[0])
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill="toself", name="Current setup"))
    fig.update_layout(title=title, polar=dict(radialaxis=dict(visible=True, range=[0, 1]), bgcolor="rgba(0,0,0,0)"), showlegend=False, margin=dict(l=30, r=30, t=60, b=30))
    return fig


def build_score_bar_figure(problem_scores: dict[str, float], title: str = "Business Problem Pressure") -> go.Figure:
    safe_scores = _finite_score_dict(problem_scores)
    labels = list(safe_scores.keys())
    values = [float(safe_scores[k]) for k in labels]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=["#8a1c7c", "#c46b00", "#008060"]))
    fig.update_layout(title=title, yaxis=dict(range=[0, 1]), margin=dict(l=30, r=30, t=60, b=30))
    return fig


def comparison_table(current_bundle: RecommendationBundle, baseline_bundle: RecommendationBundle) -> pd.DataFrame:
    current = current_bundle.ranked_df[[c for c in ["article_id", "rank", "final_score", "prod_name", "category"] if c in current_bundle.ranked_df.columns]].copy()
    baseline = baseline_bundle.ranked_df[[c for c in ["article_id", "rank", "final_score", "prod_name", "category"] if c in baseline_bundle.ranked_df.columns]].copy()
    current = current.rename(columns={"rank": "current_rank", "final_score": "current_score", "prod_name": "current_name", "category": "current_category"})
    baseline = baseline.rename(columns={"rank": "baseline_rank", "final_score": "baseline_score", "prod_name": "baseline_name", "category": "baseline_category"})
    comparison = current.merge(baseline, on="article_id", how="outer")
    comparison["rank_shift"] = pd.to_numeric(comparison.get("baseline_rank"), errors="coerce") - pd.to_numeric(comparison.get("current_rank"), errors="coerce")
    comparison["score_delta"] = pd.to_numeric(comparison.get("current_score"), errors="coerce") - pd.to_numeric(comparison.get("baseline_score"), errors="coerce")
    return comparison.sort_values(["current_rank", "baseline_rank"], na_position="last")


def recommendation_card_html(row: pd.Series, explanation: str | None = None) -> str:
    explanation = explanation or ""
    title = _safe_text(row.get("prod_name") or row.get("product_type_name") or row.get("article_id"))
    category = _safe_text(row.get("category") or row.get("product_group_name") or "unknown")
    detail_desc = _safe_text(str(row.get("detail_desc", ""))[:180])
    if not detail_desc:
        detail_desc = _safe_text(_display_description(row)[:180])
    rank = row.get("rank", "-")
    score = float(row.get("final_score", 0.0))
    deadstock = float(row.get("is_deadstock_proxy", row.get("eco_tag", 0.0)) or 0.0)
    margin = row.get("item_margin_tier_proxy", row.get("margin_score", "n/a"))
    return_risk = row.get("return_probability", 0.0)
    age_days = row.get("article_recency_days", row.get("stock_age_days", "n/a"))
    thumb = _thumbnail_token(row)
    return f"""
    <div class='eco-card'>
      <div class='eco-card-head'>
        <div class='eco-thumb'>{thumb}</div>
        <div>
          <span class='eco-rank'>#{rank}</span>
          <span class='eco-title'>{title}</span>
        </div>
      </div>
      <div class='eco-sub'>{category}</div>
      <div class='eco-desc'>{detail_desc if detail_desc else 'No description available.'}</div>
      <div class='eco-metadata'>
        <span><b>Score</b> {score:.3f}</span>
        <span><b>Margin</b> {margin}</span>
        <span><b>Deadstock</b> {deadstock:.2f}</span>
        <span><b>Return risk</b> {float(return_risk):.2f}</span>
        <span><b>Last sold days</b> {age_days}</span>
      </div>
      <div class='eco-expl'>{_safe_text(explanation)}</div>
    </div>
    """


def build_card_explanation(row: pd.Series, active_strategies: Sequence[str], strategy_weights: dict[str, float], sustainable_enabled: bool) -> str:
    title = str(row.get("prod_name") or row.get("product_type_name") or row.get("article_id"))
    primary_reason, _secondary_reason = _explanation_reasons(row, active_strategies, strategy_weights)
    if not sustainable_enabled:
        return f"{title} stays strong on core relevance for this shopper."
    return f"{title} rises because {primary_reason}."


def build_comparison_card_html(title: str, bundle: RecommendationBundle, accent: str, subtitle: str) -> str:
    metrics = bundle.problem_scores
    return f"""
    <div class='eco-compare-card' style='--eco-accent:{accent};'>
      <div class='eco-compare-head'>
        <div>
          <div class='eco-compare-eyebrow'>{_safe_text(subtitle)}</div>
          <div class='eco-compare-title'>{_safe_text(title)}</div>
        </div>
        <div class='eco-compare-pill'>{_safe_text(bundle.summary.get('mode', 'n/a'))}</div>
      </div>
      <div class='eco-compare-grid'>
        <div><span>Coverage gap</span><strong>{metrics.get('catalog_coverage_gap', 0.0):.3f}</strong></div>
        <div><span>Popularity bias</span><strong>{metrics.get('popularity_bias', 0.0):.3f}</strong></div>
        <div><span>Trust gap</span><strong>{metrics.get('transparency_trust_gap', 0.0):.3f}</strong></div>
        <div><span>Top-K</span><strong>{len(bundle.ranked_df)}</strong></div>
      </div>
    </div>
    """


def _top_popularity_table(ranked_df: pd.DataFrame) -> pd.DataFrame:
    frame = ranked_df.copy()
    if frame.empty:
        return frame
    frame["popularity"] = pd.to_numeric(_series_or_default(frame, "popularity", 0.0), errors="coerce").fillna(0.0)
    frame["category"] = _series_or_default(frame, "category", "unknown").fillna("unknown").astype(str)
    frame["popularity_rank"] = frame["popularity"].rank(ascending=False, method="dense")
    return frame[["article_id", "category", "final_score", "popularity", "popularity_rank"]].head(10)


def _controls_signature(controls: dict[str, Any]) -> tuple[Any, ...]:
    active = tuple(sorted([str(s) for s in controls.get("active_strategies", [])]))
    weights = tuple(sorted([(str(k), round(float(v), 4)) for k, v in dict(controls.get("strategy_weights", {})).items()]))
    return (
        str(controls.get("mode", "")),
        str(controls.get("use_case", "")),
        str(controls.get("user_id", "")),
        int(controls.get("top_k", 6)),
        bool(controls.get("gemma_enabled", False)),
        bool(controls.get("sustainable_enabled", False)),
        active,
        weights,
        str(controls.get("cold_start_query") or ""),
    )


def _get_cached_bundle(assets: DemoAssets, controls: dict[str, Any]) -> RecommendationBundle:
    cache_key = _controls_signature(controls)
    cache = st.session_state.setdefault("bundle_cache", {})
    if cache_key in cache:
        return cache[cache_key]
    bundle = recommend_user_bundle(
        assets=assets,
        user_id=controls["user_id"],
        mode=controls["mode"],
        use_case=controls["use_case"],
        top_k=controls["top_k"],
        sustainable_enabled=controls["sustainable_enabled"],
        active_strategies=controls["active_strategies"],
        strategy_weights=controls["strategy_weights"],
        gemma_enabled=controls["gemma_enabled"],
        cold_start_query=controls["cold_start_query"],
    )
    if len(cache) >= 20:
        cache.pop(next(iter(cache)))
    cache[cache_key] = bundle
    st.session_state["bundle_cache"] = cache
    return bundle


def _compute_bundle_with_telemetry(assets: DemoAssets, controls: dict[str, Any], bundle_label: str) -> tuple[RecommendationBundle, dict[str, Any]]:
    if not str(controls.get("user_id", "")).strip():
        raise ValueError(f"{bundle_label}: missing user_id in controls.")
    cache_key = _controls_signature(controls)
    cache = st.session_state.setdefault("bundle_cache", {})
    cache_hit = cache_key in cache
    t0 = time.perf_counter()
    bundle = _get_cached_bundle(assets, controls)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return bundle, {
        "label": bundle_label,
        "cache_hit": bool(cache_hit),
        "latency_ms": float(latency_ms),
        "top_k": int(controls.get("top_k", 0) or 0),
        "use_case": str(controls.get("use_case", "")),
        "mode": str(controls.get("mode", "")),
    }


def _render_runtime_error(message: str, exc: Exception) -> None:
    st.error(message)
    st.exception(exc)
    st.info("Try Reset in the sidebar. If the issue persists, click 'Clear recommendation cache' below and rerun.")
    if st.button("Clear recommendation cache", use_container_width=False):
        st.session_state["bundle_cache"] = {}
        st.rerun()


def _sidebar_controls(assets: DemoAssets) -> dict[str, Any]:
    st.sidebar.markdown("### Control Room")
    st.sidebar.caption("Controls update instantly in the sidebar. Click Apply to commit them to the body. Reset restores defaults.")

    demo_user_candidates = assets.user_ids[:30]
    user_labels = {str(uid): assets.users_by_id[str(uid)]["label"] for uid in demo_user_candidates}
    pending_key = "control_pending_state"
    applied_key = "control_applied_state"
    defaults = {
        "mode": "baseline",
        "use_case": "item_item_cf",
        "user_id": str(demo_user_candidates[0]) if demo_user_candidates else "",
        "top_k": 6,
        "gemma_enabled": False,
        "active_strategies": list(DEFAULT_WEIGHTS.keys()),
        "strategy_weights": dict(DEFAULT_WEIGHTS),
        "cold_start_query": "",
    }
    if pending_key not in st.session_state:
        st.session_state[pending_key] = dict(defaults)
    if applied_key not in st.session_state:
        st.session_state[applied_key] = dict(st.session_state[pending_key])

    pending_state = dict(st.session_state[pending_key])
    if pending_state.get("user_id") not in [str(u) for u in demo_user_candidates]:
        pending_state["user_id"] = str(demo_user_candidates[0]) if demo_user_candidates else ""
    st.session_state[pending_key] = pending_state

    strategies = list(DEFAULT_WEIGHTS.keys())
    if "pending_use_case" not in st.session_state:
        st.session_state["pending_use_case"] = pending_state.get("use_case", "item_item_cf")
    if "pending_mode" not in st.session_state:
        st.session_state["pending_mode"] = pending_state.get("mode", "baseline")
    if "pending_user_id" not in st.session_state:
        st.session_state["pending_user_id"] = pending_state.get("user_id", str(demo_user_candidates[0]) if demo_user_candidates else "")
    if "pending_top_k" not in st.session_state:
        st.session_state["pending_top_k"] = int(pending_state.get("top_k", 6))
    if "pending_gemma" not in st.session_state:
        st.session_state["pending_gemma"] = bool(pending_state.get("gemma_enabled", False))
    if "pending_active_strategies" not in st.session_state:
        st.session_state["pending_active_strategies"] = [s for s in pending_state.get("active_strategies", strategies) if s in strategies]
    if "pending_cold_query" not in st.session_state:
        st.session_state["pending_cold_query"] = pending_state.get("cold_start_query", "") or ""
    for name, default_weight in DEFAULT_WEIGHTS.items():
        weight_key = f"pending_weight_{name}"
        if weight_key not in st.session_state:
            st.session_state[weight_key] = float(pending_state.get("strategy_weights", {}).get(name, default_weight))

    st.sidebar.markdown("**1. Base recommender**")
    use_case = st.sidebar.selectbox("Base recommender", list(USE_CASE_LABELS.keys()), format_func=lambda x: USE_CASE_LABELS.get(x, x), help="Selects the source model that creates the initial candidate list before any reranking.", key="pending_use_case")
    st.sidebar.caption("This choice creates the first ranked candidate list and baseline relevance scores.")

    uses_coldstart = use_case == "coldstart_content_fallback"
    if uses_coldstart:
        st.session_state["pending_mode"] = "baseline"

    st.sidebar.markdown("**2. Final ranking mode**")
    mode_options = ["baseline"] if uses_coldstart else ["baseline", "sustainable"]
    mode = st.sidebar.radio("Final ranking mode", mode_options, help="Baseline keeps model ranking. Sustainable applies strategy-aware reranking.", disabled=uses_coldstart, key="pending_mode")
    if uses_coldstart:
        st.sidebar.caption("Cold-start fallback uses baseline-only ranking; sustainable reranking is disabled.")

    st.sidebar.markdown("**3. Demo context**")
    current_user_choices = [str(u) for u in demo_user_candidates]
    if st.session_state.get("pending_user_id") not in current_user_choices and current_user_choices:
        st.session_state["pending_user_id"] = current_user_choices[0]
    user_id = st.sidebar.selectbox("Demo user", demo_user_candidates, format_func=lambda uid: user_labels.get(str(uid), f"User | {str(uid)[:8]}"), help="Diversified test-user sample. Body applies these settings only after Apply.", key="pending_user_id")
    selected_user_id = str(user_id) if user_id is not None else ""
    top_k = int(st.sidebar.slider("Top-K results", min_value=4, max_value=12, step=1, help="Controls how many items are shown after ranking.", key="pending_top_k"))
    gemma_enabled = st.sidebar.toggle("Show Gemma explanation", help="Uses Gemini API when configured, otherwise falls back to local explanation text.", key="pending_gemma")

    strategy_controls_enabled = mode == "sustainable" and not uses_coldstart
    st.sidebar.markdown("### Strategy Controls")
    st.sidebar.caption("Strategy controls are active. Apply commits them to the body." if strategy_controls_enabled else "Strategy controls are inactive for baseline mode or cold-start fallback.")
    active_strategies = st.sidebar.multiselect("Active strategies", strategies, disabled=not strategy_controls_enabled, help="Pick strategy signals for sustainable reranking.", key="pending_active_strategies")

    strategy_weights: dict[str, float] = {}
    with st.sidebar.expander("Strategy weights", expanded=True):
        if strategy_controls_enabled and active_strategies:
            for name in active_strategies:
                strategy_weights[name] = float(st.slider(name.replace("_", " ").title(), min_value=0.0, max_value=0.35, step=0.01, disabled=False, help=STRATEGY_TOOLTIPS.get(name, "Strategy weight influence in sustainable reranking."), key=f"pending_weight_{name}"))
            normalized = normalize_weights({name: float(strategy_weights.get(name, 0.0)) for name in active_strategies})
            formula = "final_score = " + f"{normalized.get('baseline_relevance', 0.0):.2f} * baseline_relevance"
            for name in active_strategies:
                formula += f" + {normalized.get(name, 0.0):.2f} * score_{name}"
            st.caption("Final score formula (normalized):")
            st.code(formula, language="text")
        elif strategy_controls_enabled:
            st.caption("Select at least one active strategy to configure weights and see the final-score formula.")
        else:
            st.caption("No strategy sliders in baseline/cold-start mode.")

    if uses_coldstart:
        if not str(st.session_state.get("pending_cold_query", "")).strip():
            st.session_state["pending_cold_query"] = build_cold_start_query(assets, selected_user_id)
        cold_start_query = st.sidebar.text_area("Cold-start query text", help="Used only in cold-start mode. It applies after clicking Apply.", key="pending_cold_query")
    else:
        cold_start_query = None

    st.sidebar.caption(f"Gemini configured: {'yes' if get_gemini_client() is not None else 'no'} | Model: {_get_secret('GEMINI_MODEL') or DEFAULT_GEMINI_MODEL}")

    pending_now = {
        "mode": mode,
        "use_case": use_case,
        "user_id": selected_user_id,
        "top_k": top_k,
        "gemma_enabled": gemma_enabled,
        "active_strategies": active_strategies,
        "strategy_weights": strategy_weights,
        "cold_start_query": cold_start_query,
    }
    st.session_state[pending_key] = pending_now
    col_apply, col_reset = st.sidebar.columns(2)
    apply_clicked = col_apply.button("Apply", use_container_width=True)
    reset_clicked = col_reset.button("Reset", use_container_width=True)

    if reset_clicked:
        st.session_state[pending_key] = dict(defaults)
        st.session_state[applied_key] = dict(defaults)
        for key_name in ["pending_use_case", "pending_mode", "pending_user_id", "pending_top_k", "pending_gemma", "pending_active_strategies", "pending_cold_query"]:
            if key_name in st.session_state:
                del st.session_state[key_name]
        for name in DEFAULT_WEIGHTS.keys():
            weight_key = f"pending_weight_{name}"
            if weight_key in st.session_state:
                del st.session_state[weight_key]
        st.rerun()

    if apply_clicked:
        st.session_state[applied_key] = dict(st.session_state[pending_key])

    final_state = dict(st.session_state[applied_key])
    final_use_case = final_state.get("use_case", "item_item_cf")
    final_mode = final_state.get("mode", "baseline")
    final_sustainable_enabled = final_mode == "sustainable" and final_use_case != "coldstart_content_fallback"
    return {
        "mode": final_mode,
        "use_case": final_use_case,
        "user_id": str(final_state.get("user_id", "")),
        "top_k": int(final_state.get("top_k", 6)),
        "gemma_enabled": bool(final_state.get("gemma_enabled", False)),
        "sustainable_enabled": final_sustainable_enabled,
        "active_strategies": list(final_state.get("active_strategies", strategies)),
        "strategy_weights": dict(final_state.get("strategy_weights", DEFAULT_WEIGHTS)),
        "cold_start_query": final_state.get("cold_start_query"),
    }


def _render_help_tab() -> None:
    st.markdown(
        """
        <div class='help-hero'>
            <h3 style='margin:0 0 0.25rem 0;'>Complete Dashboard Documentation</h3>
            <p style='margin:0;color:#eef5ff;'>
            This Help tab is designed as an in-app project manual for EcoNudge. It explains the real-world problem,
            the recommender architecture, the formulas behind ranking, the business and sustainability logic, the
            evaluation metrics, and the explainability layer for both technical and non-technical readers.
            </p>
            <div>
                <span class='help-pill'>Controls</span>
                <span class='help-pill'>Backend Logic</span>
                <span class='help-pill'>Formulas</span>
                <span class='help-pill'>Charts</span>
                <span class='help-pill'>Interpretation</span>
                <span class='help-pill'>Project Context</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    doc_tabs = st.tabs(["Overview", "Controls", "Architecture", "Metrics", "Strategies", "Explainability", "Limits"])

    with doc_tabs[0]:
        st.markdown("<div class='help-section-title'>1) Project Overview</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='help-card'>
                <h4>What EcoNudge solves</h4>
                <p>
                EcoNudge is a sustainable fashion recommendation system built around a real-world tension:
                standard recommenders optimize relevance and clicks, but in fast fashion this can worsen deadstock,
                overexpose popular items, and indirectly increase waste. EcoNudge addresses that by adding a transparent
                reranking layer on top of standard recommenders so user relevance, business performance, and
                sustainability goals can be balanced explicitly.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class='help-card'>
                <h4>Why this project matters</h4>
                <p>
                This project is meant to show more than a recommendation demo. It demonstrates how recommender systems can
                be framed as real-world decision systems that affect inventory waste, customer trust, return behavior,
                logistics efficiency, and the visibility of long-tail products. The dashboard makes those trade-offs visible
                instead of hiding them inside a black-box score.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        summary_df = pd.DataFrame(
            [
                {"Area": "Domain", "Detail": "Fashion e-commerce recommendation using the H&M personalization dataset"},
                {"Area": "Core problem", "Detail": "Popularity-driven ranking hides long-tail stock and worsens inventory waste"},
                {"Area": "Project response", "Detail": "Combine baseline relevance with sustainability-aware reranking signals"},
                {"Area": "UN SDG alignment", "Detail": "SDG 12: Responsible Consumption and Production"},
                {"Area": "Prototype status", "Detail": "Working research prototype with data pipeline, baseline models, reranking logic, evaluation, and explainability layer"},
            ]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.caption("Portfolio takeaway: EcoNudge is positioned as a responsible recommendation case study where ranking quality, business logic, and sustainability logic are evaluated together.")

        st.markdown("<div class='help-section-title'>2) Use Cases</div>", unsafe_allow_html=True)
        use_case_df = pd.DataFrame(
            [
                {
                    "Use case": "Item-Item CF baseline",
                    "Purpose": "Recommend items similar to what users already interacted with",
                    "Strength": "Fast, intuitive collaborative baseline",
                    "Limitation": "Can amplify popularity bias and underexpose the long tail",
                },
                {
                    "Use case": "Implicit feedback confidence",
                    "Purpose": "Use interaction confidence to strengthen collaborative relevance",
                    "Strength": "Handles repeated implicit interactions better than plain CF",
                    "Limitation": "Still inherits historical behavior bias from observed data",
                },
                {
                    "Use case": "Cold-start content fallback",
                    "Purpose": "Recommend from item text when user-item interaction evidence is weak",
                    "Strength": "Useful for sparse-user or sparse-item settings",
                    "Limitation": "Runs in baseline-only mode and does not use the full sustainable reranking path",
                },
            ]
        )
        st.dataframe(use_case_df, use_container_width=True, hide_index=True)

        st.markdown("<div class='help-section-title'>3) Business Pillars and Business Problems</div>", unsafe_allow_html=True)
        pillars_df = pd.DataFrame(
            [
                {"Pillar / Problem": "Relevance", "Why it matters": "Users should still receive recommendations that fit their interests."},
                {"Pillar / Problem": "Novelty", "Why it matters": "The system should surface less obvious items instead of only repeating mainstream winners."},
                {"Pillar / Problem": "Diversity", "Why it matters": "Recommendation lists should not collapse into one narrow category or style."},
                {"Pillar / Problem": "Coverage", "Why it matters": "More of the catalog should receive visibility, especially long-tail inventory."},
                {"Pillar / Problem": "Explainability", "Why it matters": "Users should understand why an item appears, especially when sustainability nudges are involved."},
                {"Pillar / Problem": "Popularity bias", "Why it matters": "Purely click-driven ranking keeps reinforcing already-exposed items."},
                {"Pillar / Problem": "Deadstock pressure", "Why it matters": "Unsold aging inventory is both a cost problem and a sustainability problem."},
                {"Pillar / Problem": "Return risk", "Why it matters": "Returns create operational waste, reverse logistics cost, and environmental burden."},
                {"Pillar / Problem": "Logistics burden", "Why it matters": "Longer or less efficient fulfillment can increase emissions and system cost."},
            ]
        )
        st.dataframe(pillars_df, use_container_width=True, hide_index=True)

        st.info(
            "High-level interpretation: EcoNudge is not trying to replace relevance. It is trying to make trade-offs visible and controllable so the recommender can support users, the business, and sustainability objectives at the same time."
        )

    with doc_tabs[1]:
        st.markdown("<div class='help-section-title'>4) Controls and Their Backend Effect</div>", unsafe_allow_html=True)
        controls_df = pd.DataFrame(
            [
                {"Control": "Base recommender", "User action": "Choose CF / implicit / cold-start", "Backend effect": "Selects the source candidate list and baseline relevance scores.", "What to watch": "Recommendation quality, candidate pool shape, cold-start behavior"},
                {"Control": "Final ranking mode", "User action": "Switch baseline or sustainable", "Backend effect": "Turns the reranking layer on or off.", "What to watch": "Rank shift, score delta, popularity bias, trust gap"},
                {"Control": "Demo user", "User action": "Choose one test user profile", "Backend effect": "Loads a different shopper context and precomputed candidate pool.", "What to watch": "User profile fit, category mix, explanation relevance"},
                {"Control": "Top-K", "User action": "Change result size", "Backend effect": "Limits output length and affects coverage-based metrics.", "What to watch": "Coverage, diversity, chart interpretation"},
                {"Control": "Show Gemma explanation", "User action": "Turn explanations on or off", "Backend effect": "Calls Gemini when configured; otherwise uses safe local fallback logic.", "What to watch": "Explainability, trust gap, explanation style"},
                {"Control": "Active strategies", "User action": "Select strategy subset", "Backend effect": "Only selected strategies contribute to final scoring.", "What to watch": "Which business or sustainability objective dominates"},
                {"Control": "Strategy sliders", "User action": "Assign weights", "Backend effect": "Controls the normalized contribution of each strategy in final_score.", "What to watch": "Novelty, popularity bias, rank shift, relevance stability"},
                {"Control": "Apply / Reset", "User action": "Commit or restore state", "Backend effect": "Apply computes bundles; Reset restores default control values.", "What to watch": "Telemetry and comparison consistency"},
            ]
        )
        st.dataframe(controls_df, use_container_width=True, hide_index=True)
        st.markdown(
            """
            <div class='help-card'>
                <h4>How to read the controls as a story</h4>
                <p>
                The sidebar is intentionally structured like a decision funnel. First you pick the source recommender,
                then you decide whether to preserve its ranking or rerank it for sustainability, and only then do you tune
                which strategic objective should dominate. This makes the dashboard useful not only for experimentation,
                but also for explaining recommendation governance to stakeholders.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='help-section-title'>5) How to Use the Dashboard</div>", unsafe_allow_html=True)
        st.code(
            """1. Choose a base recommender
2. Pick baseline or sustainable mode
3. Select a demo user
4. Adjust Top-K and explanation toggle
5. Activate strategies and set weights
6. Click Apply
7. Read Control Room, User Demo, and Comparison together

Recommended reading order:
- Control Room: understand current configuration and score mix
- User Demo: inspect recommendations and explanation narrative
- Comparison: evaluate trade-offs versus baseline
""",
            language="text",
        )
        st.info("Important behavior: cold-start mode forces baseline ranking. Sustainable reranking is intentionally disabled in that path.")

    with doc_tabs[2]:
        st.markdown("<div class='help-section-title'>6) End-to-End Backend Logic</div>", unsafe_allow_html=True)
        st.code(
            """Step 1: Load lightweight precomputed live artifact
Step 2: Read controls from the sidebar
Step 3: Select one use-case-specific candidate pool for the chosen user
Step 4: Compute baseline ranking or sustainable reranking
Step 5: Build an active bundle and a baseline comparison bundle
Step 6: Compute metrics, charts, and tables
Step 7: Generate explanation text when explanation mode is enabled
Step 8: Render Control Room, User Demo, Comparison, and Help tabs
""",
            language="text",
        )

        arch_df = pd.DataFrame(
            [
                {"Layer": "Data engineering", "Role": "Create processed item, customer, transaction, and evaluation artifacts from the H&M data sample."},
                {"Layer": "Baseline models", "Role": "Produce user-item relevance using item-item collaborative filtering and matrix factorization."},
                {"Layer": "Candidate generation", "Role": "Collect the top candidate set before reranking."},
                {"Layer": "Strategy scoring", "Role": "Compute deadstock, margin, return-risk, logistics, and loyalty signals."},
                {"Layer": "Fusion", "Role": "Blend baseline relevance with strategy contributions using normalized weights."},
                {"Layer": "Evaluation", "Role": "Measure ranking quality and beyond-accuracy behavior such as novelty and coverage."},
                {"Layer": "Explainability", "Role": "Generate a user-facing explanation from structured item and strategy evidence."},
            ]
        )
        st.dataframe(arch_df, use_container_width=True, hide_index=True)
        st.markdown(
            """
            <div class='help-card'>
                <h4>Architecture design principle</h4>
                <p>
                The system is intentionally modular. Relevance is learned in the baseline layer, strategic value is added
                in the reranking layer, and user-facing trust is handled in the explanation layer. That separation matters:
                it keeps trade-offs inspectable and makes the system easier to reason about than an opaque end-to-end model.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='help-section-title'>7) Methodology Details from the Research Prototype</div>", unsafe_allow_html=True)
        methodology_df = pd.DataFrame(
            [
                {"Detail": "Dataset", "Value": "H&M Personalization Challenge data"},
                {"Detail": "Original scale", "Value": "31M transactions, 2.6M users, 105K articles"},
                {"Detail": "Prototype sample", "Value": "750K sampled transactions for tractable experimentation"},
                {"Detail": "Split", "Value": "Chronological split on t_dat with 80/20 train/test and minimum two interactions per user"},
                {"Detail": "Item-item baseline", "Value": "Cosine similarity over user-item interactions"},
                {"Detail": "Matrix factorization baseline", "Value": "Truncated SVD with 32 latent factors, seed 42"},
                {"Detail": "Evaluation cohort", "Value": "200 users in the main offline evaluation workflow"},
            ]
        )
        st.dataframe(methodology_df, use_container_width=True, hide_index=True)
        st.markdown(
            """
            <div class='help-card'>
                <h4>Live prototype serving scope</h4>
                <p>
                The public-facing live version does not ship the heavy training runtime. Instead, it serves a compact artifact
                built from the research outputs. The current live bundle contains 50 representative users and expanded
                precomputed candidate pools so the dashboard can show more recommendation variation while remaining fast enough
                for cloud deployment.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.warning(
            "This live app is a lightweight serving layer, not the full training environment. The heavy models and full parquet artifacts are replaced with compact precomputed candidate pools so the prototype can run online."
        )

    with doc_tabs[3]:
        st.markdown("<div class='help-section-title'>8) Ranking Formula and Strategy Formulas</div>", unsafe_allow_html=True)
        st.code(
            """final_score = w_baseline * baseline_relevance + sum_i(w_i * score_strategy_i)

Normalization logic:
positive = max(0, user_weight)
total = sum(positive)

if total < 1.0:
    w_baseline = 1.0 - total
else:
    w_baseline = 0.0
    strategy weights are rescaled to sum to 1.0
""",
            language="text",
        )
        formula_df = pd.DataFrame(
            [
                {"Strategy": "Deadstock Savior", "Signal source": "stock_age_days / article_recency_days", "Formula": "minmax(stock_age_days)", "Business goal": "Reduce unsold aging inventory"},
                {"Strategy": "Margin UpSell", "Signal source": "margin_score and eco_tag", "Formula": "minmax(0.7 * margin_score + 0.3 * eco_tag)", "Business goal": "Support profitability with an eco-aware blend"},
                {"Strategy": "Return Killer", "Signal source": "return_probability", "Formula": "minmax(1 - return_probability)", "Business goal": "Reduce returns and reverse logistics burden"},
                {"Strategy": "Logistics Bundler", "Signal source": "distance_km", "Formula": "minmax(exp(-distance_km / 300))", "Business goal": "Favor lighter logistics paths"},
                {"Strategy": "Loyalty Multiplier", "Signal source": "loyalty_affinity", "Formula": "minmax(loyalty_affinity)", "Business goal": "Keep relevance stable for repeat users"},
            ]
        )
        st.dataframe(formula_df, use_container_width=True, hide_index=True)
        st.markdown(
            """
            <div class='help-card'>
                <h4>What the formula means in practice</h4>
                <p>
                The final score is not a replacement for relevance. It is a weighted negotiation between relevance and
                strategic objectives. A higher baseline weight means the system behaves more like a standard recommender.
                A lower baseline weight means the dashboard is acting more like a policy instrument that actively steers
                exposure toward inventory, logistics, trust, or sustainability goals.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='help-section-title'>9) Metrics, Charts, and Interpretation</div>", unsafe_allow_html=True)
        metrics_df = pd.DataFrame(
            [
                {"Metric / Chart": "Novelty", "How it is computed here": "mean(1 - normalized popularity)", "How to read it": "Higher means the list leans more toward less-exposed items."},
                {"Metric / Chart": "Diversity", "How it is computed here": "unique categories / item_count", "How to read it": "Higher means more within-list category variety."},
                {"Metric / Chart": "Serendipity", "How it is computed here": "mean((1 - popularity_norm) * normalized final_score)", "How to read it": "Higher means items are both competitive and less obvious."},
                {"Metric / Chart": "Coverage", "How it is computed here": "recommended unique items / catalog size", "How to read it": "Higher means the system exposes more of the catalog."},
                {"Metric / Chart": "Popularity bias", "How it is computed here": "rank-weighted popularity concentration", "How to read it": "Lower is better if the goal is to reduce overexposure of already-popular items."},
                {"Metric / Chart": "Trust gap", "How it is computed here": "1 - explainability score", "How to read it": "Lower means recommendation reasoning is more available and visible."},
                {"Metric / Chart": "Recommendation delta table", "How it is computed here": "baseline rank/score versus current rank/score", "How to read it": "Positive rank_shift means the item moved upward under sustainable reranking."},
                {"Metric / Chart": "Popularity and coverage view", "How it is computed here": "top recommendation popularity by category", "How to read it": "Use it to see whether the ranking spreads attention or remains concentrated."},
            ]
        )
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.markdown(
            """
            <div class='help-card'>
                <h4>How to interpret changes responsibly</h4>
                <p>
                No single metric should be read in isolation. A strong sustainability run may improve novelty while hurting
                relevance stability. A commercially strong run may improve efficiency but raise popularity bias. The purpose
                of the dashboard is not to produce one perfect number, but to make these trade-offs explicit and auditable.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("How to interpret metric trade-offs", expanded=False):
            st.markdown(
                """
                - Higher novelty is often good for deadstock recovery, but it can reduce stable relevance if pushed too hard.
                - Lower popularity bias usually signals better long-tail exposure, but it may conflict with highly safe commercial items.
                - Coverage is naturally tiny in a small Top-K setting, so compare relative shifts rather than raw absolute values.
                - Trust gap should drop when explanation support is active and successful.
                - If sustainable reranking changes the order but not the membership of Top-K, some metrics may move only slightly.
                """
            )

    with doc_tabs[4]:
        st.markdown("<div class='help-section-title'>10) Sustainability Strategies and Expected Effects</div>", unsafe_allow_html=True)
        strategy_df = pd.DataFrame(
            [
                {
                    "Strategy": "Deadstock Savior",
                    "Problem it addresses": "Warehouse accumulation and long-tail invisibility",
                    "What increasing weight usually does": "Pushes older inventory upward more aggressively",
                    "Likely metric effects": "Novelty often rises, popularity bias may fall, relevance can become less stable if overused",
                },
                {
                    "Strategy": "Margin UpSell",
                    "Problem it addresses": "Commercial viability and assortment profitability",
                    "What increasing weight usually does": "Promotes stronger-margin candidates with an eco-aware blend",
                    "Likely metric effects": "Popularity bias can rise if profitable items are also mainstream; novelty may flatten",
                },
                {
                    "Strategy": "Return Killer",
                    "Problem it addresses": "Costly returns and reverse logistics",
                    "What increasing weight usually does": "Favors lower return-risk candidates",
                    "Likely metric effects": "Operational safety improves, but serendipity can shrink if the system becomes too conservative",
                },
                {
                    "Strategy": "Logistics Bundler",
                    "Problem it addresses": "Shipping burden and fulfillment efficiency",
                    "What increasing weight usually does": "Pushes more logistics-efficient items upward",
                    "Likely metric effects": "Operational sustainability improves, but coverage may not improve much because this is an efficiency strategy, not an exploration strategy",
                },
                {
                    "Strategy": "Loyalty Multiplier",
                    "Problem it addresses": "User trust and repeat-user relevance",
                    "What increasing weight usually does": "Keeps rankings closer to known repeat-interest patterns",
                    "Likely metric effects": "Perceived relevance improves, but novelty and diversity can decline if user history is narrow",
                },
            ]
        )
        st.dataframe(strategy_df, use_container_width=True, hide_index=True)
        st.markdown(
            """
            <div class='help-card'>
                <h4>How to think about strategy tuning</h4>
                <p>
                The strategies are not meant to be maxed out simultaneously. Each one encodes a different stakeholder
                priority. The most realistic usage pattern is to increase one objective deliberately, observe what moves in
                the comparison tab, and then decide whether the resulting trade-off is acceptable from a user, business,
                and sustainability perspective.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='help-section-title'>11) Business Pillars and Sustainable Ranking Philosophy</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='help-card'>
                <h4>Design philosophy</h4>
                <p>
                EcoNudge uses a weighted multi-objective fusion approach. Baseline relevance remains the protective anchor,
                while strategy weights create controlled sustainability and business nudges. The intention is not to coerce
                the user away from relevance, but to gently rebalance what becomes visible.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code(
            """Interpretation shortcut:
- More baseline weight = safer, more relevance-preserving ranking
- More deadstock weight = stronger inventory recovery signal
- More margin weight = stronger business optimization
- More return/logistics weight = stronger operational sustainability
- More loyalty weight = stronger repeat-user personalization
""",
            language="text",
        )

    with doc_tabs[5]:
        st.markdown("<div class='help-section-title'>12) Gemma / Gemini Explainability Layer</div>", unsafe_allow_html=True)
        explain_df = pd.DataFrame(
            [
                {"Component": "Top explanation box", "Purpose": "Provide one shopper-facing explanation for the current top recommendation"},
                {"Component": "Card explanations", "Purpose": "Provide shorter per-item reasoning tied to strategy effects"},
                {"Component": "Gemini API", "Purpose": "Generate polished natural-language explanations when API access is configured"},
                {"Component": "Fallback logic", "Purpose": "Provide deterministic local explanations when the API is unavailable or disabled"},
            ]
        )
        st.dataframe(explain_df, use_container_width=True, hide_index=True)
        st.markdown(
            """
            <div class='help-card'>
                <h4>Explainability design goal</h4>
                <p>
                Explanations in EcoNudge are not decorative text. They are part of the accountability layer. Because the
                ranking can be influenced by business and sustainability objectives, the system should give users and
                reviewers a readable reason for why an item is being surfaced. The live prototype therefore treats fallback
                explanations as a safety mechanism, not as a failure state.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class='help-card'>
                <h4>Why explainability matters here</h4>
                <p>
                In EcoNudge, recommendations are not only relevance-driven. Some items are elevated because they help solve
                business or sustainability problems such as deadstock, return risk, or logistics burden. Without explanations,
                this becomes a black box. With explanations, the system exposes the trade-off instead of hiding it.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.code(
            """Explanation behavior in the live prototype:
- If 'Show Gemma explanation' is ON and Gemini is configured, the app attempts a generated explanation
- If Gemini is unavailable or fails, the app falls back to rule-based local explanation logic
- If the toggle is OFF, explainability should not be counted as active in the metrics
""",
            language="text",
        )

        with st.expander("Technical note: why fallback exists", expanded=False):
            st.markdown(
                """
                The research project explicitly treats fallback as a safety mechanism. A live portfolio app should never
                hallucinate false reasons for why an item was shown. That is why deterministic fallback remains part of the
                architecture even when a language model is available.
                """
            )

    with doc_tabs[6]:
        st.markdown("<div class='help-section-title'>13) Evidence, Limitations, and Future Work</div>", unsafe_allow_html=True)
        limits_df = pd.DataFrame(
            [
                {"Area": "Data scale", "Current state": "Research code references the full H&M challenge scale, but the working prototype uses a 750K sample for practicality."},
                {"Area": "Evaluation", "Current state": "Results are offline and directional, not a live A/B test."},
                {"Area": "Environmental signals", "Current state": "Deadstock, logistics, and return signals are proxies, not full life-cycle measurements."},
                {"Area": "Explainability", "Current state": "Generated explanations are useful, but they are still a narrative layer on top of structured signals."},
                {"Area": "Live app architecture", "Current state": "The portfolio app uses compact precomputed artifacts rather than the full training runtime."},
            ]
        )
        st.dataframe(limits_df, use_container_width=True, hide_index=True)
        st.markdown(
            """
            <div class='help-card'>
                <h4>What to claim confidently</h4>
                <p>
                This project demonstrates a working sustainable recommender prototype, a complete experimentation pipeline,
                interpretable reranking logic, and a dashboard that exposes ranking trade-offs clearly. It does not claim a
                finished production deployment or perfect causal measurement of environmental impact. That distinction makes
                the portfolio stronger, because it shows engineering maturity and honest evaluation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='help-section-title'>14) What This Prototype Demonstrates in a Portfolio Context</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='help-card'>
                <h4>Why this matters as a real-world case study</h4>
                <p>
                EcoNudge is not just a ranking demo. It shows how recommender systems can be designed as decision systems
                that make stakeholder trade-offs explicit. The project addresses real operational problems:
                deadstock, popularity bias, returns, logistics burden, explainability, and the conflict between revenue and sustainability.
                That makes it portfolio-relevant both as a machine learning system and as a product-thinking case study.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.code(
            """Future improvement directions:
- validate sustainability proxies with stronger external evidence
- expand fairness and segment-level evaluation
- improve explanation grounding with richer structured evidence
- add business guardrails and scenario presets
- move from offline simulation toward deployment experiments
""",
            language="text",
        )

        st.success(
            "This Help tab is intended to function as a compact project handbook: it summarizes the problem, the recommender logic, the formulas, the strategy trade-offs, the evaluation perspective, and the explainability layer in one place."
        )


def main() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
    try:
        assets = load_assets(_bundle_mtime())
    except Exception as exc:
        st.markdown(
            """
            <div class='eco-hero'>
                <h1>EcoNudge: Explainable Sustainable Fashion Recommender</h1>
                <p>The dashboard cannot load the demo assets in the current Python environment.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.error(str(exc))
        st.info("Use the live app environment and ensure live/assets/demo_bundle.json exists.")
        st.stop()

    controls = _sidebar_controls(assets)
    comparison_controls = dict(controls)
    comparison_controls["mode"] = "baseline"
    comparison_controls["sustainable_enabled"] = False
    comparison_controls["gemma_enabled"] = False

    st.markdown(
        """
        <div class='eco-hero'>
            <div class='eco-brand'>
                <div class='eco-logo'>EN</div>
                <div class='eco-brand-copy'>
                    <h1>EcoNudge: Explainable Sustainable Fashion Recommender</h1>
                    <p>Explainable e-commerce recommendation dashboard balancing relevance, sustainability, margin, returns, logistics, and trust.</p>
                </div>
            </div>
            <div>
                <span class='eco-badge'>Use cases</span>
                <span class='eco-badge'>Business pillars</span>
                <span class='eco-badge'>Business problems</span>
                <span class='eco-badge'>Sustainability strategies</span>
                <span class='eco-badge'>Gemma explainability</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.text(
        "Explainable sustainable fashion recommender for e-commerce with interactive strategy controls, reranking, business metrics, and sustainability-aware recommendations."
    )

    tabs = st.tabs(["Control Room", "User Demo", "Comparison", "Help"])

    with st.spinner("Applying controls and computing recommendations..."):
        try:
            bundle, current_telemetry = _compute_bundle_with_telemetry(assets, controls, "active_bundle")
            baseline_bundle, baseline_telemetry = _compute_bundle_with_telemetry(assets, comparison_controls, "baseline_bundle")
        except Exception as exc:
            _render_runtime_error("Recommendation computation failed. The app stopped to avoid partial or inconsistent output.", exc)
            st.stop()

    total_compute_ms = float(current_telemetry["latency_ms"]) + float(baseline_telemetry["latency_ms"])
    consecutive_stuck_runs = int(st.session_state.get("consecutive_stuck_runs", 0))
    consecutive_stuck_runs = consecutive_stuck_runs + 1 if total_compute_ms >= STUCK_COMPUTE_THRESHOLD_MS else 0
    st.session_state["consecutive_stuck_runs"] = consecutive_stuck_runs
    if total_compute_ms >= SLOW_COMPUTE_THRESHOLD_MS:
        st.warning(f"Recommendation compute is slower than target ({round(total_compute_ms, 1)} ms). Try reducing Top-K or switching to baseline mode.")
    if consecutive_stuck_runs >= STUCK_CONSECUTIVE_RUNS:
        st.error("Recommendation compute appears repeatedly stuck/slow. Open the error details below if a failure occurs, then clear cache and retry.")

    st.sidebar.markdown("### Live score snapshot")
    st.sidebar.caption(
        f"Compute telemetry: active {round(float(current_telemetry['latency_ms']), 1)} ms "
        f"({'cache' if bool(current_telemetry['cache_hit']) else 'fresh'}), "
        f"baseline {round(float(baseline_telemetry['latency_ms']), 1)} ms "
        f"({'cache' if bool(baseline_telemetry['cache_hit']) else 'fresh'})"
    )
    st.sidebar.plotly_chart(build_score_bar_figure(bundle.problem_scores, title="Sidebar score snapshot"), use_container_width=True)
    st.sidebar.caption("This view tracks how the current configuration shifts catalog coverage, popularity bias, and trust pressure.")

    with tabs[0]:
        st.markdown("### Live Configuration")
        summary_df = pd.DataFrame(
            [
                {"setting": "mode", "value": controls["mode"]},
                {"setting": "use_case", "value": controls["use_case"]},
                {"setting": "user_id", "value": controls["user_id"]},
                {"setting": "gemma_enabled", "value": controls["gemma_enabled"]},
                {"setting": "sustainable_enabled", "value": controls["sustainable_enabled"]},
                {"setting": "active_strategies", "value": ", ".join(controls["active_strategies"])},
            ]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        left, right = st.columns([1.1, 0.9])
        with left:
            st.plotly_chart(build_radar_figure(bundle.radar_scores, title="Current pillar score radar"), use_container_width=True)
        with right:
            st.plotly_chart(build_score_bar_figure(bundle.problem_scores, title="Business problem pressure"), use_container_width=True)
        st.markdown("#### Strategy impact")
        impact_df = pd.DataFrame([bundle.summary]).assign(
            catalog_coverage_gap=bundle.problem_scores["catalog_coverage_gap"],
            popularity_bias=bundle.problem_scores["popularity_bias"],
            transparency_trust_gap=bundle.problem_scores["transparency_trust_gap"],
        )
        st.dataframe(impact_df, use_container_width=True, hide_index=True)
        st.markdown("#### Reranking Layer (Manager View)")
        if controls["sustainable_enabled"] and controls["active_strategies"]:
            normalized = normalize_weights({name: float(controls["strategy_weights"].get(name, 0.0)) for name in controls["active_strategies"]})
            formula = "final_score = " + f"{normalized.get('baseline_relevance', 0.0):.2f} * baseline_relevance"
            for name in controls["active_strategies"]:
                formula += f" + {normalized.get(name, 0.0):.2f} * score_{name}"
            st.code(formula, language="text")
            weights_df = pd.DataFrame([{"component": "baseline_relevance", "weight": float(normalized.get("baseline_relevance", 0.0))}] + [{"component": name, "weight": float(normalized.get(name, 0.0))} for name in controls["active_strategies"]])
            st.dataframe(weights_df, use_container_width=True, hide_index=True)
            st.caption("Weights are normalized before scoring. If strategy weights sum to less than 1, the remainder stays on baseline relevance.")
        else:
            st.info("Reranking layer is inactive. Final score currently follows baseline relevance only.")

    with tabs[1]:
        st.markdown("### Logged-in User View")
        user_profile = build_user_profile(assets, controls["user_id"])
        col_a, col_b = st.columns([0.45, 0.55])
        with col_a:
            st.markdown("#### User profile")
            st.dataframe(pd.DataFrame([user_profile]), use_container_width=True, hide_index=True)
            st.markdown("#### Login context")
            st.info(f"User {controls['user_id']} is viewing the catalog in {controls['mode']} mode using {controls['use_case']}.")
            if bundle.explanation_text:
                st.markdown("#### Gemma explanation")
                st.success(bundle.explanation_text)
        with col_b:
            st.plotly_chart(build_radar_figure(bundle.radar_scores, title="User-specific recommendation radar"), use_container_width=True)
        st.markdown("#### Recommended products")
        cards = bundle.ranked_df.copy()
        for start in range(0, len(cards), 2):
            row_chunk = cards.iloc[start:start + 2]
            columns = st.columns(len(row_chunk))
            for col, (_, row) in zip(columns, row_chunk.iterrows()):
                with col:
                    card_explanation = build_card_explanation(row, controls["active_strategies"], controls["strategy_weights"], controls["sustainable_enabled"])
                    st.markdown(recommendation_card_html(row, card_explanation), unsafe_allow_html=True)
        st.markdown("#### Recommendation table")
        display_cols = [c for c in ["rank", "article_id", "prod_name", "category", "detail_desc", "final_score", "baseline_relevance", "item_margin_tier_proxy", "is_deadstock_proxy", "tx_return_risk_mean", "tx_logistics_burden_mean", "article_recency_days"] if c in bundle.ranked_df.columns]
        st.dataframe(bundle.ranked_df[display_cols], use_container_width=True, hide_index=True)

    with tabs[2]:
        st.markdown("### Baseline vs Sustainable comparison")
        left_cmp, right_cmp = st.columns(2)
        with left_cmp:
            st.markdown(build_comparison_card_html("Baseline", baseline_bundle, "#c46b00", "Raw ranking only"), unsafe_allow_html=True)
            st.plotly_chart(build_radar_figure(baseline_bundle.radar_scores, title="Baseline radar"), use_container_width=True)
            st.plotly_chart(build_score_bar_figure(baseline_bundle.problem_scores, title="Baseline pressure"), use_container_width=True)
        with right_cmp:
            st.markdown(build_comparison_card_html("Sustainable", bundle, "#008060", "Strategy-aware ranking"), unsafe_allow_html=True)
            st.plotly_chart(build_radar_figure(bundle.radar_scores, title="Sustainable radar"), use_container_width=True)
            st.plotly_chart(build_score_bar_figure(bundle.problem_scores, title="Sustainable pressure"), use_container_width=True)
        comparison_df = comparison_table(bundle, baseline_bundle)
        comparison_cols = [c for c in ["article_id", "current_rank", "baseline_rank", "rank_shift", "current_score", "baseline_score", "score_delta", "current_name", "baseline_name", "current_category", "baseline_category"] if c in comparison_df.columns]
        st.markdown("#### Recommendation delta table")
        st.dataframe(comparison_df[comparison_cols], use_container_width=True, hide_index=True)
        st.markdown("#### Popularity and coverage view")
        top_pop = _top_popularity_table(bundle.ranked_df)
        if not top_pop.empty:
            fig = go.Figure()
            for category in top_pop["category"].astype(str).unique().tolist():
                subset = top_pop[top_pop["category"].astype(str) == category]
                fig.add_trace(go.Bar(name=category, x=subset["article_id"], y=subset["popularity"]))
            fig.update_layout(barmode="group", title="Recommended item popularity profile")
            st.plotly_chart(fig, use_container_width=True)
        readout_df = pd.DataFrame([{"mode": "baseline", **baseline_bundle.problem_scores}, {"mode": "sustainable", **bundle.problem_scores}])
        st.markdown("#### Strategic readout")
        st.dataframe(readout_df, use_container_width=True, hide_index=True)

    with tabs[3]:
        _render_help_tab()

    st.caption("EcoNudge demo prototype. Control room changes are reflected live in the user-facing recommendation view.")


if __name__ == "__main__":
    main()
