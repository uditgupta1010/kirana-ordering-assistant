"""Kirana Ordering Assistant - Streamlit Application

An educational prototype for an FMCG distributor supplying 300 kirana stores.
Reads WhatsApp order messages in Hinglish, matches to product catalog via RAG,
asks follow-ups for ambiguous items, predicts reorder nudges with an ML model,
and enforces a human checkpoint before order dispatch.

Core Principle: The LLM proposes, Python validates.
Product IDs, prices, and totals always come from the catalog, never from the LLM.
"""

from __future__ import annotations
import os
import json
import time
import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Kirana Ordering Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Core imports
from guardrails import (
    load_ordering_rules,
    load_message_templates,
    render_template,
    load_catalog,
    compute_usual_quantities,
    check_quantity_flag,
    calculate_line_total,
    is_prompt_injection,
)
from matching import CatalogMatcher
from extraction import (
    DEFAULT_MODEL,
    BACKUP_MODEL,
    get_groq_client,
    discover_available_models,
    extract_order_lines,
    choose_candidates_with_llm,
    get_cache_key,
)
from reorder_model import (
    load_order_history,
    build_feature_dataset,
    train_model,
    evaluate_baselines_and_model,
    compute_feature_importance,
    run_stress_test,
    predict_store_nudges,
)


# ==============================================================================
# CSS STYLING (Inter + DM Mono, Navy headers, WhatsApp Green, Amber flags)
# Bulletproof contrast: readable on all screens and system dark/light modes.
# ==============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

/* Force clean light background on app container */
.stApp {
    background-color: #f8f7f4 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #0f1f3d !important;
}

/* Ensure all markdown text, labels, and spans are high-contrast navy */
.stApp p, .stApp span, .stApp label, .stApp li,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] span,
div[data-testid="stMarkdownContainer"] li {
    color: #0f1f3d !important;
    font-size: 0.92rem;
}

/* Header typography */
h1, h2, h3, h4, h5, h6,
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] h4 {
    color: #0f1f3d !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
}

/* Monospace for SKU codes, numbers, and currency amounts */
.mono-text, .sku-badge, .rupee-amount {
    font-family: 'DM Mono', monospace !important;
}

/* Sidebar styling: solid white background and clear text */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
section[data-testid="stSidebar"] * {
    color: #0f1f3d !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: #0f1f3d !important;
}

/* Tabs: crisp and readable */
button[data-baseweb="tab"] {
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 10px 16px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #0f1f3d !important;
    border-bottom: 3px solid #0f1f3d !important;
    font-weight: 700 !important;
}

/* WhatsApp styled chat messages container */
.chat-container {
    background-color: #efeae2 !important;
    background-image: radial-gradient(#d1d5db 1px, transparent 1px) !important;
    background-size: 16px 16px !important;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #cbd5e1;
    margin-bottom: 16px;
    min-height: 200px;
}

/* Metric KPI card */
.kpi-card {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-bottom: 12px;
}
.kpi-title {
    font-size: 0.8rem !important;
    text-transform: uppercase;
    color: #475569 !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em;
}
.kpi-value {
    font-family: 'DM Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #0f1f3d !important;
    margin-top: 4px;
}
.kpi-sub {
    font-size: 0.78rem !important;
    color: #64748b !important;
    margin-top: 2px;
}

/* Amber alert badge for flags */
.amber-flag {
    background-color: #fef3c7 !important;
    color: #92400e !important;
    border: 1px solid #f59e0b !important;
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    margin: 8px 0;
}
.amber-flag * {
    color: #92400e !important;
}

/* Card container for risks & checkpoints */
.risk-card {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-left: 5px solid #2563eb !important;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.risk-card p, .risk-card strong, .risk-card div, .risk-card code {
    color: #0f1f3d !important;
}
.risk-card.danger {
    border-left-color: #dc2626 !important;
}
.risk-card.warning {
    border-left-color: #d97706 !important;
}
.risk-card.success {
    border-left-color: #059669 !important;
}

/* Dataframe and Data Editor crisp readability */
div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
    width: 100% !important;
    overflow-x: auto !important;
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
}

/* Form inputs, selectboxes, and text inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > input,
div[data-testid="stChatInput"] textarea {
    color: #0f1f3d !important;
    background-color: #ffffff !important;
}

/* ================================================================
   NUCLEAR OVERRIDES — maximum specificity to beat Streamlit dark mode
   ================================================================ */

/* ALL buttons: force white bg + navy text */
html body .stApp .stButton > button,
html body .stApp button[data-testid="baseButton-secondary"],
html body .stApp button[data-testid="stBaseButton-secondary"] {
    background-color: #ffffff !important;
    color: #0f1f3d !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
html body .stApp .stButton > button:hover,
html body .stApp button[data-testid="baseButton-secondary"]:hover {
    background-color: #f1f5f9 !important;
    color: #0f1f3d !important;
    border-color: #94a3b8 !important;
}
html body .stApp .stButton > button p,
html body .stApp .stButton > button span,
html body .stApp .stButton > button div {
    color: #0f1f3d !important;
}

/* PRIMARY buttons: blue bg + white text */
html body .stApp button[data-testid="baseButton-primary"],
html body .stApp .stButton > button[kind="primary"] {
    background-color: #1e3a8a !important;
    color: #ffffff !important;
    border: 1px solid #1e3a8a !important;
}
html body .stApp button[data-testid="baseButton-primary"]:hover {
    background-color: #1e40af !important;
    color: #ffffff !important;
}
html body .stApp button[data-testid="baseButton-primary"] p,
html body .stApp button[data-testid="baseButton-primary"] span,
html body .stApp button[data-testid="baseButton-primary"] div {
    color: #ffffff !important;
}

/* Sidebar buttons: light grey bg */
html body section[data-testid="stSidebar"] .stButton > button,
html body section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {
    background-color: #f1f5f9 !important;
    color: #0f1f3d !important;
    border: 1px solid #cbd5e1 !important;
}
html body section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #e2e8f0 !important;
}
html body section[data-testid="stSidebar"] .stButton > button p,
html body section[data-testid="stSidebar"] .stButton > button span {
    color: #0f1f3d !important;
}

/* Risk cards — FORCE white background */
html body .stApp .risk-card,
html body .stApp div.risk-card {
    background-color: #ffffff !important;
    color: #0f1f3d !important;
}
html body .stApp .risk-card *,
html body .stApp div.risk-card * {
    color: #0f1f3d !important;
}
html body .stApp .risk-card h4 {
    color: inherit !important;
}
html body .stApp .risk-card.warning h4 { color: #92400e !important; }
html body .stApp .risk-card.danger h4 { color: #991b1b !important; }
html body .stApp .risk-card.success h4 { color: #065f46 !important; }

/* KPI cards — FORCE white background */
html body .stApp .kpi-card,
html body .stApp div.kpi-card {
    background-color: #ffffff !important;
    color: #0f1f3d !important;
}
html body .stApp .kpi-card * {
    color: inherit !important;
}
html body .stApp .kpi-title {
    color: #475569 !important;
}
html body .stApp .kpi-value {
    color: #0f1f3d !important;
}
html body .stApp .kpi-sub {
    color: #64748b !important;
}

/* Inline <code> elements: always visible */
html body .stApp code,
html body .stApp .risk-card code,
html body .stApp .kpi-card code {
    background-color: #f1f5f9 !important;
    color: #0f1f3d !important;
    padding: 2px 6px !important;
    border-radius: 3px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85em !important;
}

/* Streamlit metric widget */
html body .stApp div[data-testid="stMetric"],
html body .stApp div[data-testid="stMetricValue"],
html body .stApp div[data-testid="metric-container"] {
    background-color: #ffffff !important;
    color: #0f1f3d !important;
}
html body .stApp div[data-testid="stMetricValue"] *,
html body .stApp div[data-testid="stMetricLabel"] * {
    color: #0f1f3d !important;
}

/* Expander header & content */
html body .stApp div[data-testid="stExpander"] summary,
html body .stApp div[data-testid="stExpander"] summary *,
html body .stApp div[data-testid="stExpander"] div {
    color: #0f1f3d !important;
}

/* Selectbox / dropdown */
html body .stApp div[data-baseweb="select"] * {
    color: #0f1f3d !important;
}

/* Radio & checkbox labels */
html body .stApp div[data-testid="stRadio"] label,
html body .stApp div[data-testid="stCheckbox"] label {
    color: #0f1f3d !important;
}

/* Toast / alert messages */
html body .stApp div[data-testid="stAlert"],
html body .stApp div[data-testid="stAlert"] * {
    color: #0f1f3d !important;
}

/* Chat messages */
html body .stApp div[data-testid="stChatMessage"] * {
    color: #0f1f3d !important;
}

/* Amber flag badge */
html body .stApp .amber-flag,
html body .stApp .amber-flag * {
    background-color: #fef3c7 !important;
    color: #92400e !important;
}


/* ================================================================
   FIXES (Streamlit 1.46 test ids): primary buttons, chat input, checkbox
   ================================================================ */

/* PRIMARY buttons: blue bg + white text (enabled, hover, disabled) */
html body .stApp .stButton > button[kind="primary"],
html body .stApp button[data-testid="stBaseButton-primary"] {
    background-color: #1e3a8a !important;
    color: #ffffff !important;
    border: 1px solid #1e3a8a !important;
}
html body .stApp .stButton > button[kind="primary"]:hover,
html body .stApp button[data-testid="stBaseButton-primary"]:hover {
    background-color: #1e40af !important;
    border-color: #1e40af !important;
    color: #ffffff !important;
}
html body .stApp .stButton > button[kind="primary"] p,
html body .stApp .stButton > button[kind="primary"] span,
html body .stApp .stButton > button[kind="primary"] div,
html body .stApp button[data-testid="stBaseButton-primary"] p,
html body .stApp button[data-testid="stBaseButton-primary"] span,
html body .stApp button[data-testid="stBaseButton-primary"] div {
    color: #ffffff !important;
}
html body .stApp .stButton > button[kind="primary"]:disabled,
html body .stApp button[data-testid="stBaseButton-primary"]:disabled {
    background-color: #8fa0c4 !important;
    border-color: #8fa0c4 !important;
    color: #ffffff !important;
    cursor: not-allowed !important;
    opacity: 1 !important;
}
html body .stApp .stButton > button[kind="primary"]:disabled p,
html body .stApp .stButton > button[kind="primary"]:disabled span,
html body .stApp .stButton > button[kind="primary"]:disabled div {
    color: #ffffff !important;
}

/* Chat input: white pill with a light border, readable placeholder, visible send icon */
html body .stApp div[data-testid="stChatInput"] > div,
html body .stApp div[data-testid="stChatInput"] div[data-baseweb="textarea"],
html body .stApp div[data-testid="stChatInput"] div[data-baseweb="base-input"] {
    background-color: #ffffff !important;
}
html body .stApp div[data-testid="stChatInput"] > div {
    border: 1px solid #cbd5e1 !important;
    border-radius: 20px !important;
    box-shadow: 0 1px 3px rgba(15, 31, 61, 0.08) !important;
}
html body .stApp div[data-testid="stChatInput"] > div:focus-within {
    border-color: #1e3a8a !important;
}
html body .stApp div[data-testid="stChatInput"] textarea {
    color: #0f1f3d !important;
    background-color: #ffffff !important;
    caret-color: #0f1f3d !important;
}
html body .stApp div[data-testid="stChatInput"] textarea::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}
html body .stApp div[data-testid="stChatInput"] div[data-testid="InputInstructions"],
html body .stApp div[data-testid="stChatInput"] div[data-testid="InputInstructions"] * {
    color: #64748b !important;
}
html body .stApp button[data-testid="stChatInputSubmitButton"] {
    background-color: transparent !important;
    color: #1e3a8a !important;
}
html body .stApp button[data-testid="stChatInputSubmitButton"]:hover {
    color: #0f1f3d !important;
}
html body .stApp button[data-testid="stChatInputSubmitButton"]:disabled {
    color: #94a3b8 !important;
}

/* Checkbox: white box with grey border when unchecked, navy when checked */
html body .stApp div[data-testid="stCheckbox"] label[data-baseweb="checkbox"] > span:first-of-type {
    background-color: #ffffff !important;
    border: 2px solid #94a3b8 !important;
    border-radius: 4px !important;
}
html body .stApp div[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:hover > span:first-of-type {
    border-color: #1e3a8a !important;
}
html body .stApp div[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:checked) > span:first-of-type,
html body .stApp div[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input[aria-checked="true"]) > span:first-of-type {
    background-color: #1e3a8a !important;
    border-color: #1e3a8a !important;
}
html body .stApp div[data-testid="stCheckbox"] label[data-baseweb="checkbox"] p {
    color: #0f1f3d !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# SECRETS & RESOURCE LOADING WITH DEFENSIVE GUARDS
# ==============================================================================

def verify_and_get_api_key() -> str:
    """Verifies Groq API key from st.secrets defensively."""
    key = None
    try:
        if "groq_api_key" in st.secrets:
            key = st.secrets["groq_api_key"]
    except Exception:
        pass
        
    if not key or key == "PASTE_YOUR_GROQ_KEY_HERE" or not str(key).strip():
        st.error(
            "⚠️ **Groq API Key Missing or Invalid**\n\n"
            "Please paste your Groq API key into `.streamlit/secrets.toml`:\n"
            "```toml\n"
            "groq_api_key = \"gsk_...\"\n"
            "```\n"
            "On Streamlit Community Cloud, enter `groq_api_key` under **App Settings > Secrets**."
        )
        st.stop()
    return str(key).strip()


@st.cache_resource(show_spinner=False)
def load_cached_catalog_and_matcher() -> Tuple[Dict[str, Any], CatalogMatcher]:
    try:
        catalog = load_catalog("data/catalog.json")
        matcher = CatalogMatcher(catalog)
        return catalog, matcher
    except Exception as e:
        st.error(f"⚠️ **Error loading catalog**: {e}")
        st.stop()


@st.cache_resource(show_spinner=False)
def load_and_train_reorder_model() -> Tuple[pd.DataFrame, Any, pd.DataFrame, pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    try:
        df_hist = load_order_history("data/order_history.csv")
        feat_df = build_feature_dataset(df_hist)
        model, train_df, test_df = train_model(feat_df, random_state=42)
        eval_metrics = evaluate_baselines_and_model(model, test_df)
        stress_res = run_stress_test(model, test_df)
        imp_df = compute_feature_importance(model, test_df, random_state=42)
        return df_hist, model, train_df, test_df, stress_res, imp_df
    except Exception as e:
        st.error(f"⚠️ **Error training reorder model**: {e}")
        st.stop()


# ==============================================================================
# INITIALIZATION & STARTUP CHECKS
# ==============================================================================

# Verify key
groq_key = verify_and_get_api_key()
groq_client = get_groq_client(groq_key)

# Read rule files fresh on every run (never cached)
try:
    rules_settings, rules_text = load_ordering_rules("ordering-rules.md")
    templates = load_message_templates("message-templates.md")
except Exception as e:
    st.error(f"⚠️ **Error reading rule or template files**: {e}")
    st.stop()

# Load catalog & ML model
catalog, matcher = load_cached_catalog_and_matcher()
df_history, reorder_clf, train_df, test_df, stress_results, importance_df = load_and_train_reorder_model()
store_sku_medians, catalog_medians = compute_usual_quantities(df_history)

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_order_items" not in st.session_state:
    st.session_state.current_order_items = []
if "unresolved_followups" not in st.session_state:
    st.session_state.unresolved_followups = []
if "current_not_in_catalog" not in st.session_state:
    st.session_state.current_not_in_catalog = []
if "current_nudges" not in st.session_state:
    st.session_state.current_nudges = []
if "declined_nudges" not in st.session_state:
    st.session_state.declined_nudges = []
if "confirmed_orders" not in st.session_state:
    st.session_state.confirmed_orders = []
if "llm_cache" not in st.session_state:
    st.session_state.llm_cache = {}
if "selected_store_id" not in st.session_state:
    st.session_state.selected_store_id = "ST012"
if "order_confirmed" not in st.session_state:
    st.session_state.order_confirmed = False
if "made_up_attempts_count" not in st.session_state:
    st.session_state.made_up_attempts_count = 0


# Startup check: verify store ST238 triggers oil nudge
def startup_nudge_verification():
    oil_skus = ["SKU009", "SKU010", "SKU011"]
    oil_preds = predict_store_nudges(
        store_id="ST238",
        df_history=df_history,
        clf=reorder_clf,
        catalog_products=matcher.sku_to_product,
        current_order_skus=["SKU005", "SKU034"],  # Hide & Seek and MDH Mirch
        threshold=rules_settings.get("nudge_probability_threshold", 0.7),
        max_nudges=2,
    )
    nudge_sku_ids = [p["sku_id"] for p in oil_preds]
    if not any(s in nudge_sku_ids for s in oil_skus):
        st.warning(
            "⚠️ Startup Check: Store ST238 did not trigger an oil nudge with current threshold. "
            "Please check nudge_probability_threshold in ordering-rules.md."
        )

startup_nudge_verification()


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================

st.sidebar.markdown("### 🛒 Kirana Assistant")
st.sidebar.markdown("*Distributor Ordering Portal*")

# Store selector
stores_unique = (
    df_history[["store_id", "store_name", "store_size"]]
    .drop_duplicates()
    .sort_values("store_id")
)
store_options = [
    f"{r['store_id']} - {r['store_name']} ({r['store_size']})"
    for _, r in stores_unique.iterrows()
]
store_ids = stores_unique["store_id"].tolist()

# Find current index
current_store_idx = 0
if st.session_state.selected_store_id in store_ids:
    current_store_idx = store_ids.index(st.session_state.selected_store_id)

selected_store_str = st.sidebar.selectbox(
    "Select Kirana Store:",
    options=store_options,
    index=current_store_idx,
    help="Select the shopkeeper sending WhatsApp orders.",
)
selected_store_id = selected_store_str.split(" - ")[0]

# Detect store change and reset chat
if selected_store_id != st.session_state.selected_store_id:
    st.session_state.selected_store_id = selected_store_id
    st.session_state.chat_history = []
    st.session_state.current_order_items = []
    st.session_state.unresolved_followups = []
    st.session_state.current_not_in_catalog = []
    st.session_state.current_nudges = []
    st.session_state.declined_nudges = []
    st.session_state.order_confirmed = False
    st.rerun()

store_meta = stores_unique[stores_unique["store_id"] == selected_store_id].iloc[0]
store_name = store_meta["store_name"]
store_size = store_meta["store_size"]

st.sidebar.markdown("---")

# Demo Case Quick Buttons
st.sidebar.markdown("#### 🎯 Quick Demo Cases")
demo_cases_file = "data/demo_cases.json"
demo_cases = []
if os.path.exists(demo_cases_file):
    with open(demo_cases_file) as f:
        demo_cases = json.load(f).get("cases", [])

def load_demo(case_id: str):
    case = next((c for c in demo_cases if c["id"] == case_id), None)
    if not case:
        return
    st.session_state.selected_store_id = case["store_id"]
    st.session_state.chat_history = []
    st.session_state.current_order_items = []
    st.session_state.unresolved_followups = []
    st.session_state.current_not_in_catalog = []
    st.session_state.current_nudges = []
    st.session_state.declined_nudges = []
    st.session_state.order_confirmed = False
    st.session_state.pending_message_to_send = case["message"]
    st.rerun()

col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    if st.button("1. Clean Order", use_container_width=True):
        load_demo("clean")
    if st.button("3. Reorder Nudge", use_container_width=True):
        load_demo("nudge")
with col_d2:
    if st.button("2. Unclear Order", use_container_width=True):
        load_demo("ambiguous")
    if st.button("4. Not In Catalog", use_container_width=True):
        load_demo("edge")

st.sidebar.markdown("---")

# Model Selection
st.sidebar.markdown("#### 🤖 LLM Model")
discovered_models = discover_available_models(groq_client)
model_options = discovered_models if discovered_models else [DEFAULT_MODEL, BACKUP_MODEL]

selected_model = st.sidebar.selectbox(
    "Active Groq Model:",
    options=model_options,
    index=0,
    help="Default is llama-3.3-70b-versatile. Automatically falls back if rate-limited.",
)

st.sidebar.markdown("---")
# Reset & Clear buttons
if st.sidebar.button("🔄 Reset Conversation", use_container_width=True):
    st.session_state.chat_history = []
    st.session_state.current_order_items = []
    st.session_state.unresolved_followups = []
    st.session_state.current_not_in_catalog = []
    st.session_state.current_nudges = []
    st.session_state.declined_nudges = []
    st.session_state.order_confirmed = False
    st.rerun()

if st.sidebar.button("🧹 Clear Cached Results", use_container_width=True):
    st.session_state.llm_cache = {}
    if os.path.exists("cache/eval_results.json"):
        try:
            os.remove("cache/eval_results.json")
        except Exception:
            pass
    st.sidebar.success("Cache cleared!")
    st.rerun()


# ==============================================================================
# MAIN PAGE TABS
# ==============================================================================

# Two core tabs
tab_chat, tab_staff = st.tabs([
    "💬 A. Shop Chat (WhatsApp)",
    "📋 B. Staff: Confirmed Orders",
])


# ==============================================================================
# TAB A: SHOP CHAT (WHATSAPP-STYLE)
# ==============================================================================

with tab_chat:
    # WhatsApp Header Card
    st.markdown(
        f"""
        <div style="background-color: #075e54; color: white; padding: 12px 18px; border-radius: 8px 8px 0 0; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong style="font-size: 1.05rem;">{store_name}</strong>
                <span class="sku-badge" style="background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; margin-left: 8px;">{selected_store_id}</span>
                <div style="font-size: 0.75rem; opacity: 0.9; margin-top: 2px;">
                    Size: <strong>{store_size.capitalize()}</strong> | Status: 🟢 Online
                </div>
            </div>
            <div style="font-size: 0.8rem; text-align: right; opacity: 0.9;">
                WhatsApp Business API
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Chat display container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        greeting_text = render_template(templates, "greeting", store_name=store_name)
        st.markdown(
            f"""
            <div style="background-color: #ffffff; padding: 10px 14px; border-radius: 8px; width: fit-content; max-width: 80%; margin-bottom: 10px; box-shadow: 0 1px 1px rgba(0,0,0,0.1);">
                <div style="font-size: 0.85rem; color: #1e293b;">{greeting_text}</div>
                <div style="font-size: 0.65rem; color: #94a3b8; text-align: right; margin-top: 4px;">Just now</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Render chat messages
    for msg in st.session_state.chat_history:
        sender = msg["sender"]
        text = msg["text"]
        timestamp = msg.get("time", "")
        
        if sender == "user":
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                    <div style="background-color: #dcf8c6; padding: 10px 14px; border-radius: 8px; width: fit-content; max-width: 80%; box-shadow: 0 1px 1px rgba(0,0,0,0.1);">
                        <div style="font-size: 0.88rem; color: #0f1f3d;">{text}</div>
                        <div style="font-size: 0.65rem; color: #64748b; text-align: right; margin-top: 4px;">{timestamp}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                    <div style="background-color: #ffffff; padding: 10px 14px; border-radius: 8px; width: fit-content; max-width: 80%; box-shadow: 0 1px 1px rgba(0,0,0,0.1);">
                        <div style="font-size: 0.88rem; color: #1e293b; white-space: pre-wrap;">{text}</div>
                        <div style="font-size: 0.65rem; color: #94a3b8; text-align: right; margin-top: 4px;">{timestamp}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --------------------------------------------------------------------------
    # Interactive Follow-up Question Resolvers
    # --------------------------------------------------------------------------
    if st.session_state.unresolved_followups and not st.session_state.order_confirmed:
        st.markdown("##### ❓ Please Clarify These Items:")
        for idx, f_item in enumerate(st.session_state.unresolved_followups):
            phrase = f_item.get("phrase", "")
            q_text = f_item.get("question", "")
            options = f_item.get("options", [])
            
            with st.container():
                st.markdown(f"**Item**: *\"{phrase}\"* — {q_text}")
                if options:
                    opt_cols = st.columns(len(options))
                    for opt_idx, opt_cand in enumerate(options):
                        with opt_cols[opt_idx]:
                            cand_name = opt_cand.get("name", str(opt_cand))
                            cand_sku = opt_cand.get("sku_id")
                            if st.button(f"👉 {cand_name}", key=f"followup_btn_{idx}_{opt_idx}"):
                                # Resolve follow-up with this candidate
                                line = f_item.get("line", {})
                                qty = line.get("qty", 1) or 1
                                unit_w = line.get("unit_word", "piece")
                                
                                c_unit, p_mult, u_ask, _ = matcher.convert_unit(qty, unit_w, cand_sku)
                                if not u_ask and cand_sku:
                                    calc = calculate_line_total(cand_sku, qty, p_mult, matcher.sku_to_product)
                                    calc["source"] = "Message (Clarified)"
                                    calc["matched_via"] = f"Clarified by Shopkeeper: {cand_name}"
                                    calc["unit"] = c_unit
                                    st.session_state.current_order_items.append(calc)
                                    st.session_state.unresolved_followups.pop(idx)
                                    
                                    # Add to chat
                                    st.session_state.chat_history.append({
                                        "sender": "user",
                                        "text": f"Bhaiya, {cand_name} bhej do",
                                        "time": datetime.datetime.now().strftime("%H:%M"),
                                    })
                                    st.session_state.chat_history.append({
                                        "sender": "assistant",
                                        "text": f"Theek hai, {cand_name} ({qty} {c_unit}) add kar diya.",
                                        "time": datetime.datetime.now().strftime("%H:%M"),
                                    })
                                    st.rerun()
                else:
                    st.caption("Awaiting shopkeeper quantity / unit reply in message.")
                    
    # --------------------------------------------------------------------------
    # Interactive Reorder Nudge Cards
    # --------------------------------------------------------------------------
    if st.session_state.current_nudges and not st.session_state.order_confirmed:
        st.markdown("##### 💡 Reorder Nudges (Frequently Ordered Items Missing):")
        for n_idx, nudge_item in enumerate(st.session_state.current_nudges):
            n_sku = nudge_item["sku_id"]
            n_name = nudge_item["name"]
            n_price = nudge_item["unit_price"]
            n_proba = nudge_item["probability"]
            nudge_msg = render_template(templates, "nudge", product=n_name)
            
            n_col1, n_col2, n_col3 = st.columns([3, 1, 1])
            with n_col1:
                st.markdown(
                    f"**{nudge_msg}**  \n"
                    f"<span style='font-size:0.8rem; color:#64748b;'>Model confidence: {n_proba * 100:.1f}% | Rs {n_price}/pc</span>",
                    unsafe_allow_html=True,
                )
            with n_col2:
                if st.button("✅ Haan, bhej do", key=f"nudge_yes_{n_idx}"):
                    # Add 1 case or usual order to order items
                    calc = calculate_line_total(n_sku, 1, nudge_item.get("case_size", 1), matcher.sku_to_product)
                    calc["source"] = "Nudge"
                    calc["matched_via"] = f"ML Nudge (Confidence {n_proba*100:.1f}%)"
                    calc["unit"] = "case"
                    st.session_state.current_order_items.append(calc)
                    st.session_state.current_nudges.pop(n_idx)
                    
                    st.session_state.chat_history.append({
                        "sender": "user",
                        "text": f"Haan, {n_name} bhi bhej do",
                        "time": datetime.datetime.now().strftime("%H:%M"),
                    })
                    st.session_state.chat_history.append({
                        "sender": "assistant",
                        "text": f"{n_name} order mein add kar diya gaya hai.",
                        "time": datetime.datetime.now().strftime("%H:%M"),
                    })
                    st.rerun()
            with n_col3:
                if st.button("❌ Nahi chahiye", key=f"nudge_no_{n_idx}"):
                    st.session_state.declined_nudges.append(n_sku)
                    st.session_state.current_nudges.pop(n_idx)
                    decline_msg = render_template(templates, "nudge_declined", product=n_name)
                    st.session_state.chat_history.append({
                        "sender": "user",
                        "text": f"Nahi, {n_name} nahi chahiye",
                        "time": datetime.datetime.now().strftime("%H:%M"),
                    })
                    st.session_state.chat_history.append({
                        "sender": "assistant",
                        "text": decline_msg,
                        "time": datetime.datetime.now().strftime("%H:%M"),
                    })
                    st.rerun()

    # --------------------------------------------------------------------------
    # Confirmation Screen & Human Checkpoint Grid
    # --------------------------------------------------------------------------
    if st.session_state.current_order_items:
        st.markdown("---")
        st.markdown("### 📝 Order Review & Human Checkpoint")
        st.markdown(
            "The LLM extracted the items, but **Python computes all totals and enforces checks**. "
            "Please review quantities and totals before confirming dispatch."
        )
        
        # Prepare data for st.data_editor
        grid_rows = []
        any_flagged = False
        
        for idx, item in enumerate(st.session_state.current_order_items):
            sku_id = item["sku_id"]
            pieces = item["pieces"]
            
            # Check 3x quantity sanity flag
            is_flagged, flag_reason, usual_qty = check_quantity_flag(
                selected_store_id,
                sku_id,
                pieces,
                store_sku_medians,
                catalog_medians,
                flag_multiple=rules_settings.get("quantity_flag_multiple", 3.0),
            )
            if is_flagged:
                any_flagged = True
                
            grid_rows.append({
                "SKU": sku_id,
                "Item": item["name"],
                "Pack": item["pack_size"],
                "Qty": int(item["qty"]),
                "Unit": item.get("unit", "piece"),
                "Pieces": int(pieces),
                "Price/piece (Rs)": float(item["unit_price"]),
                "Line total (Rs)": float(item["line_total"]),
                "Flag": "⚠️ High Qty" if is_flagged else "Normal",
                "_idx": idx,
            })
            
        df_editor = pd.DataFrame(grid_rows)
        
        # Data editor allowing shopkeeper to edit quantities or delete rows
        # SKU stays in the frame (hidden) so the confirm handler can record it.
        edited_df = st.data_editor(
            df_editor[["SKU", "Item", "Pack", "Qty", "Unit", "Pieces", "Price/piece (Rs)", "Line total (Rs)", "Flag"]],
            num_rows="dynamic",
            use_container_width=True,
            disabled=["SKU", "Item", "Pack", "Unit", "Pieces", "Price/piece (Rs)", "Line total (Rs)", "Flag"],
            column_config={"SKU": None},
            key="order_data_editor",
        )
        
        # Drop blank rows added in the grid (no catalog item behind them)
        edited_df = edited_df[edited_df["SKU"].notna()].copy()
        
        # Recompute Pieces and Line total from the edited Qty using catalog math
        if not edited_df.empty:
            base_rows = df_editor.loc[edited_df.index]
            pieces_per_unit = base_rows["Pieces"] / base_rows["Qty"].replace(0, 1)
            edited_df["Qty"] = (
                pd.to_numeric(edited_df["Qty"], errors="coerce").fillna(0).clip(lower=0).astype(int)
            )
            edited_df["Pieces"] = (edited_df["Qty"] * pieces_per_unit).round().astype(int)
            edited_df["Line total (Rs)"] = (edited_df["Pieces"] * edited_df["Price/piece (Rs)"]).round(2)
            
            # Write quantity edits back to the order items so the grid, flags and totals stay in sync
            qty_changed = False
            for _, r in edited_df.iterrows():
                item = st.session_state.current_order_items[int(base_rows.loc[r.name, "_idx"])]
                if int(item["qty"]) != int(r["Qty"]):
                    item["qty"] = int(r["Qty"])
                    item["pieces"] = int(r["Pieces"])
                    item["line_total"] = float(r["Line total (Rs)"])
                    qty_changed = True
            if qty_changed:
                st.rerun()
        
        # Recalculate totals from edited grid
        recomputed_grand_total = float(edited_df["Line total (Rs)"].sum()) if not edited_df.empty else 0.0
        
        # Display Grand Total Card
        tot_col1, tot_col2 = st.columns([2, 1])
        with tot_col1:
            st.markdown(
                f"""
                <div style="background: #ffffff; border: 2px solid #0f1f3d; padding: 14px 20px; border-radius: 8px;">
                    <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">GRAND TOTAL (FROM CATALOG MATH)</div>
                    <div class="rupee-amount" style="font-size: 1.8rem; font-weight: 700; color: #0f1f3d;">
                        Rs {recomputed_grand_total:,.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with tot_col2:
            st.markdown(
                f"""
                <div style="font-size: 0.85rem; color: #64748b; padding-top: 8px;">
                    Items: <strong>{len(edited_df)} lines</strong><br>
                    Total Pieces: <strong>{int(edited_df['Pieces'].sum()) if not edited_df.empty else 0} pcs</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
        # Display Not in Catalog Alerts
        if st.session_state.current_not_in_catalog:
            for ooc in st.session_state.current_not_in_catalog:
                not_in_cat_msg = render_template(templates, "not_in_catalog", phrase=ooc)
                st.markdown(
                    f"""
                    <div style="background-color: #fee2e2; border-left: 4px solid #ef4444; padding: 10px 14px; border-radius: 4px; margin-top: 10px; font-size: 0.88rem; color: #991b1b;">
                        ⚠️ <strong>Not In Catalog:</strong> {not_in_cat_msg}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
        # Amber flag checkbox if high quantity detected
        high_qty_approved = True
        if any_flagged:
            st.markdown(
                """
                <div class="amber-flag">
                    ⚠️ <strong>Dhayan dein:</strong> One or more items exceed 3x the store's typical historical order quantity.
                </div>
                """,
                unsafe_allow_html=True,
            )
            high_qty_approved = st.checkbox(
                "✅ I verify that the highlighted quantities are correct and intended.",
                value=False,
                key="confirm_high_qty_checkbox",
            )
            
        # Unresolved follow-ups blocking notice
        has_blocking_followups = len(st.session_state.unresolved_followups) > 0
        if has_blocking_followups:
            st.info("ℹ️ Please resolve the follow-up clarification questions above before confirming dispatch.")
            
        # Dispatch Confirmation Button
        confirm_enabled = (
            not st.session_state.order_confirmed
            and not has_blocking_followups
            and high_qty_approved
            and not edited_df.empty
        )
        
        st.markdown(f"*{render_template(templates, 'confirm_prompt')}*")
        
        if st.button("🚀 Confirm Order & Dispatch", type="primary", disabled=not confirm_enabled):
            order_id = f"ORD-{int(time.time()) % 100000:05d}"
            confirmed_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save lines to session state
            for _, r in edited_df.iterrows():
                st.session_state.confirmed_orders.append({
                    "Order ID": order_id,
                    "Time": confirmed_time,
                    "Store ID": selected_store_id,
                    "Store Name": store_name,
                    "SKU ID": r["SKU"],
                    "Item": r["Item"],
                    "Qty": r["Qty"],
                    "Unit": r["Unit"],
                    "Line total (Rs)": r["Line total (Rs)"],
                    "Source": "Message",
                    "Flag": r["Flag"],
                    "Matched Via": "Catalog Verified",
                })
                
            st.session_state.order_confirmed = True
            confirmed_msg = render_template(templates, "confirmed", order_id=order_id, total=f"{recomputed_grand_total:,.2f}")
            
            st.session_state.chat_history.append({
                "sender": "assistant",
                "text": confirmed_msg,
                "time": datetime.datetime.now().strftime("%H:%M"),
            })
            st.success(confirmed_msg)
            st.rerun()

    # --------------------------------------------------------------------------
    # Chat Input Box
    # --------------------------------------------------------------------------
    pending_msg = st.session_state.pop("pending_message_to_send", None)
    user_input = st.chat_input("Type your WhatsApp order (e.g. 'bhaiya 2 peti Parle-G 100g aur 1 peti Maggi bhej do')...")
    
    msg_to_process = pending_msg or user_input
    
    if msg_to_process:
        # Append user message
        st.session_state.chat_history.append({
            "sender": "user",
            "text": msg_to_process,
            "time": datetime.datetime.now().strftime("%H:%M"),
        })
        st.session_state.order_confirmed = False
        
        # Check injection guardrail upfront
        if is_prompt_injection(msg_to_process):
            reply_text = render_template(templates, "injection_reply")
            st.session_state.chat_history.append({
                "sender": "assistant",
                "text": reply_text,
                "time": datetime.datetime.now().strftime("%H:%M"),
            })
            st.rerun()
            
        with st.spinner("Reading your order..."):
            cache_key = get_cache_key(msg_to_process, rules_text, selected_model)
            cached_data = st.session_state.llm_cache.get(cache_key)
            
            if cached_data:
                extracted_lines = cached_data["extracted_lines"]
                llm_choices = cached_data["llm_choices"]
                actual_model = cached_data["model_used"]
            else:
                # Step 1: LLM Extraction
                extracted_lines, actual_model, used_fb = extract_order_lines(
                    msg_to_process, rules_text, groq_client, model=selected_model
                )
                
                # Step 2: RAG Retrieval
                phrases_to_retrieve = []
                for line in extracted_lines:
                    phrase = line.get("product_phrase", "").strip()
                    if phrase:
                        cands = matcher.retrieve_candidates(phrase, top_k=rules_settings.get("retrieval_top_k", 5))
                        phrases_to_retrieve.append({"phrase": phrase, "candidates": cands, "line": line})
                        
                # Step 2: LLM Choosing
                llm_choices, _ = choose_candidates_with_llm(
                    phrases_to_retrieve, rules_text, groq_client, model=actual_model
                )
                
                # Cache result
                st.session_state.llm_cache[cache_key] = {
                    "extracted_lines": extracted_lines,
                    "llm_choices": llm_choices,
                    "model_used": actual_model,
                }
                
            # Process extracted lines through Python guardrails
            new_order_items = []
            new_followups = []
            new_not_in_catalog = []
            
            for line in extracted_lines:
                phrase = line.get("product_phrase", "").strip()
                if not phrase:
                    continue
                    
                cands = matcher.retrieve_candidates(phrase, top_k=rules_settings.get("retrieval_top_k", 5))
                raw_choice = llm_choices.get(phrase, {}).get("choice", "NOT_IN_CATALOG")
                llm_q = llm_choices.get(phrase, {}).get("question")
                
                decision, chosen_sku, is_made_up, caption = matcher.validate_and_filter_choice(
                    phrase=phrase,
                    candidates=cands,
                    llm_choice=raw_choice,
                    match_threshold_high=rules_settings.get("match_threshold_high", 85.0),
                    match_threshold_low=rules_settings.get("match_threshold_low", 55.0),
                    match_margin=rules_settings.get("match_margin", 8.0),
                )
                
                if is_made_up:
                    st.session_state.made_up_attempts_count += 1
                    
                if decision == "NOT_IN_CATALOG":
                    new_not_in_catalog.append(phrase)
                elif decision == "ASK":
                    # Generate options
                    q_text = llm_q
                    if not q_text:
                        opt_names = " ya ".join(c["name"] for c in cands[:2])
                        q_text = render_template(templates, "followup_choice", phrase=phrase, options=opt_names)
                    new_followups.append({
                        "phrase": phrase,
                        "question": q_text,
                        "options": cands[:3],
                        "line": line,
                    })
                elif decision == "MATCH" and chosen_sku:
                    # Check quantity confidence
                    if line.get("qty_confidence") == "low" or line.get("qty") is None:
                        q_text = render_template(templates, "followup_quantity", phrase=phrase)
                        new_followups.append({
                            "phrase": phrase,
                            "question": q_text,
                            "options": [],
                            "line": line,
                        })
                    else:
                        # Unit conversion
                        c_unit, p_mult, u_ask, u_reason = matcher.convert_unit(
                            line.get("qty"), line.get("unit_word"), chosen_sku
                        )
                        if u_ask:
                            q_text = render_template(templates, "followup_unit", phrase=phrase, unit_word=line.get("unit_word", ""))
                            new_followups.append({
                                "phrase": phrase,
                                "question": q_text,
                                "options": [],
                                "line": line,
                            })
                        else:
                            calc = calculate_line_total(chosen_sku, line["qty"], p_mult, matcher.sku_to_product)
                            calc["source"] = "Message"
                            calc["matched_via"] = caption
                            calc["unit"] = c_unit
                            new_order_items.append(calc)
                            
            st.session_state.current_order_items = new_order_items
            st.session_state.unresolved_followups = new_followups
            st.session_state.current_not_in_catalog = new_not_in_catalog
            
            # Predict reorder nudges for items missing from current message
            current_skus = [it["sku_id"] for it in new_order_items]
            predicted_nudges = predict_store_nudges(
                store_id=selected_store_id,
                df_history=df_history,
                clf=reorder_clf,
                catalog_products=matcher.sku_to_product,
                current_order_skus=current_skus,
                threshold=rules_settings.get("nudge_probability_threshold", 0.7),
                max_nudges=rules_settings.get("max_nudges_per_order", 2),
                declined_skus=st.session_state.declined_nudges,
            )
            st.session_state.current_nudges = predicted_nudges
            
            # Formulate assistant reply text
            reply_lines = [render_template(templates, "understood")]
            if new_order_items:
                for it in new_order_items:
                    reply_lines.append(f"• {it['name']} — {it['qty']} {it['unit']} ({it['pieces']} pcs = Rs {it['line_total']:,.0f})")
            else:
                reply_lines.append("• Koi clear product match nahi hua.")
                
            st.session_state.chat_history.append({
                "sender": "assistant",
                "text": "\n".join(reply_lines),
                "time": datetime.datetime.now().strftime("%H:%M"),
            })
            st.rerun()


# ==============================================================================
# TAB B: STAFF (CONFIRMED ORDERS)
# ==============================================================================

with tab_staff:
    st.markdown("### 📋 Staff Order Fulfillment & Dispatches")
    st.caption("Orders recorded in memory session state. (Streamlit Community Cloud disk is temporary).")
    
    if not st.session_state.confirmed_orders:
        st.info("No orders confirmed yet. Submit an order on the Shop Chat screen to view dispatches here.")
    else:
        df_confirmed = pd.DataFrame(st.session_state.confirmed_orders)
        
        # Store filter
        stores_in_orders = ["All Stores"] + sorted(df_confirmed["Store ID"].unique().tolist())
        selected_filter = st.selectbox("Filter by Store:", options=stores_in_orders, index=0)
        
        if selected_filter != "All Stores":
            filtered_df = df_confirmed[df_confirmed["Store ID"] == selected_filter]
        else:
            filtered_df = df_confirmed
            
        st.dataframe(
            filtered_df[["Order ID", "Time", "Store ID", "Store Name", "SKU ID", "Item", "Qty", "Unit", "Line total (Rs)", "Source", "Flag", "Matched Via"]],
            hide_index=True,
            use_container_width=True,
        )
        
        # CSV Download Button
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Confirmed Orders CSV",
            data=csv_data,
            file_name=f"confirmed_orders_{datetime.date.today()}.csv",
            mime="text/csv",
        )
