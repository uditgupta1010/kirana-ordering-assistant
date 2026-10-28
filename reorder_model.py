"""reorder_model.py - Predictive model for reorder nudges, baselines, and stress testing.

Can be run standalone:
    python reorder_model.py
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance


STORE_SIZE_ENCODING = {"small": 0, "medium": 1, "large": 2}


def load_order_history(filepath: str = "data/order_history.csv") -> pd.DataFrame:
    """Loads order history CSV defensively."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing order history file: {filepath}")
    df = pd.read_csv(filepath)
    return df


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Builds features with ZERO leakage from target weeks.
    
    Rows: store x SKU (SKUs ordered by the store before week W) x week W (W in [5, 26]).
    Weeks 1-4 serve strictly as warm-up.
    """
    festival_map = dict(df[["week", "is_festival_week"]].drop_duplicates().values)
    store_size_map = dict(df[["store_id", "store_size"]].drop_duplicates().values)
    
    orders_set = set(zip(df["store_id"], df["sku_id"], df["week"]))
    store_skus = df.groupby("store_id")["sku_id"].unique().to_dict()
    
    rows: List[Dict[str, Any]] = []
    
    for w in range(5, 27):
        is_fest = festival_map.get(w, 0)
        for store_id, skus in store_skus.items():
            st_size = STORE_SIZE_ENCODING.get(store_size_map.get(store_id, "medium"), 1)
            for sku in skus:
                # Only consider prior weeks 1 to w-1
                past_weeks = [pw for pw in range(1, w) if (store_id, sku, pw) in orders_set]
                if not past_weeks:
                    continue  # Only SKUs the store has ordered before
                    
                order_freq = len(past_weeks) / (w - 1)
                weeks_since = w - max(past_weeks)
                target = 1 if (store_id, sku, w) in orders_set else 0
                
                # Baseline Rule 1: ordered last week
                rule1_pred = 1 if (w - 1) in past_weeks else 0
                
                # Baseline Rule 2: ordered in > 50% of past weeks
                rule2_pred = 1 if order_freq > 0.5 else 0
                
                rows.append({
                    "store_id": store_id,
                    "sku_id": sku,
                    "week": w,
                    "order_freq": order_freq,
                    "weeks_since": weeks_since,
                    "is_festival_week": is_fest,
                    "store_size": st_size,
                    "rule1_pred": rule1_pred,
                    "rule2_pred": rule2_pred,
                    "target": target,
                })
                
    return pd.DataFrame(rows)


def train_model(features_df: pd.DataFrame, random_state: int = 42) -> Tuple[HistGradientBoostingClassifier, pd.DataFrame, pd.DataFrame]:
    """Splits into train (weeks 5-20) and test (weeks 21-26) and trains HistGradientBoostingClassifier."""
    train_df = features_df[(features_df["week"] >= 5) & (features_df["week"] <= 20)]
    test_df = features_df[(features_df["week"] >= 21) & (features_df["week"] <= 26)].copy()
    
    feature_cols = ["order_freq", "weeks_since", "is_festival_week", "store_size"]
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    
    clf = HistGradientBoostingClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    
    return clf, train_df, test_df


def evaluate_baselines_and_model(
    clf: HistGradientBoostingClassifier,
    test_df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Computes Precision, Recall, and F1 on unseen weeks 21-26 for:
    - Rule 1: 'Ordered it last week'
    - Rule 2: 'Orders it in most weeks' (>50%)
    - ML Model: HistGradientBoostingClassifier
    """
    feature_cols = ["order_freq", "weeks_since", "is_festival_week", "store_size"]
    y_true = test_df["target"]
    
    # Rule 1
    r1 = test_df["rule1_pred"]
    p_r1 = precision_score(y_true, r1, zero_division=0)
    r_r1 = recall_score(y_true, r1, zero_division=0)
    f1_r1 = f1_score(y_true, r1, zero_division=0)
    
    # Rule 2
    r2 = test_df["rule2_pred"]
    p_r2 = precision_score(y_true, r2, zero_division=0)
    r_r2 = recall_score(y_true, r2, zero_division=0)
    f1_r2 = f1_score(y_true, r2, zero_division=0)
    
    # ML Model
    y_proba = clf.predict_proba(test_df[feature_cols])[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    p_ml = precision_score(y_true, y_pred, zero_division=0)
    r_ml = recall_score(y_true, y_pred, zero_division=0)
    f1_ml = f1_score(y_true, y_pred, zero_division=0)
    
    results = [
        {
            "Method": "Rule 1: Ordered last week",
            "Right when it predicts an order (Precision)": f"{p_r1 * 100:.1f}%",
            "Real orders it catches (Recall)": f"{r_r1 * 100:.1f}%",
            "Overall score (F1)": f"{f1_r1:.3f}",
            "_raw_f1": f1_r1,
        },
        {
            "Method": "Rule 2: Orders in >50% past weeks",
            "Right when it predicts an order (Precision)": f"{p_r2 * 100:.1f}%",
            "Real orders it catches (Recall)": f"{r_r2 * 100:.1f}%",
            "Overall score (F1)": f"{f1_r2:.3f}",
            "_raw_f1": f1_r2,
        },
        {
            "Method": "HistGradientBoostingClassifier (ML Model)",
            "Right when it predicts an order (Precision)": f"{p_ml * 100:.1f}%",
            "Real orders it catches (Recall)": f"{r_ml * 100:.1f}%",
            "Overall score (F1)": f"{f1_ml:.3f}",
            "_raw_f1": f1_ml,
        },
    ]
    return pd.DataFrame(results)


def compute_threshold_table(
    clf: HistGradientBoostingClassifier,
    test_df: pd.DataFrame,
    current_threshold: float = 0.7,
    thresholds: List[float] = [0.5, 0.6, 0.7, 0.8],
) -> pd.DataFrame:
    """Computes Precision, Recall, F1 and average nudges per store-week across thresholds."""
    feature_cols = ["order_freq", "weeks_since", "is_festival_week", "store_size"]
    y_true = test_df["target"]
    y_proba = clf.predict_proba(test_df[feature_cols])[:, 1]
    
    total_store_weeks = len(test_df[["store_id", "week"]].drop_duplicates())
    
    rows = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        n_nudges = pred.sum()
        nudges_per_order = n_nudges / total_store_weeks if total_store_weeks > 0 else 0
        
        is_current = abs(t - current_threshold) < 1e-4
        status = "Selected in rules (active)" if is_current else ""
        
        rows.append({
            "Threshold": f"{t:.1f}",
            "Precision": f"{p * 100:.1f}%",
            "Recall": f"{r * 100:.1f}%",
            "F1 Score": f"{f1:.3f}",
            "Avg Nudges / Store-Week": f"{nudges_per_order:.2f}",
            "Setting Status": status,
        })
    return pd.DataFrame(rows)


def compute_feature_importance(
    clf: HistGradientBoostingClassifier,
    test_df: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """Computes permutation importance on the unseen test set."""
    feature_cols = ["order_freq", "weeks_since", "is_festival_week", "store_size"]
    pretty_names = {
        "order_freq": "Order Frequency (past weeks)",
        "weeks_since": "Weeks Since Last Order",
        "is_festival_week": "Festival Week Indicator",
        "store_size": "Store Size (Capacity)",
    }
    
    perm = permutation_importance(
        clf,
        test_df[feature_cols],
        test_df["target"],
        n_repeats=5,
        random_state=random_state,
        scoring="f1",
    )
    
    rows = []
    for idx, col in enumerate(feature_cols):
        rows.append({
            "Feature": pretty_names.get(col, col),
            "Importance (F1 Drop)": float(perm.importances_mean[idx]),
            "Std Dev": float(perm.importances_std[idx]),
        })
    df_imp = pd.DataFrame(rows).sort_values("Importance (F1 Drop)", ascending=False)
    return df_imp


def run_stress_test(
    clf: HistGradientBoostingClassifier,
    test_df: pd.DataFrame,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Synthetic-data bias stress test:
    Compares model F1 on regular stores (ST001-ST270) vs irregular stores (ST271-ST300).
    Stores ST271-ST300 were generated with much higher ordering gap variance.
    """
    feature_cols = ["order_freq", "weeks_since", "is_festival_week", "store_size"]
    test_df = test_df.copy()
    test_df["y_pred"] = (clf.predict_proba(test_df[feature_cols])[:, 1] >= threshold).astype(int)
    
    irregular_stores = [f"ST{i:03d}" for i in range(271, 301)]
    reg_test = test_df[~test_df["store_id"].isin(irregular_stores)]
    irreg_test = test_df[test_df["store_id"].isin(irregular_stores)]
    
    f1_reg = f1_score(reg_test["target"], reg_test["y_pred"], zero_division=0)
    p_reg = precision_score(reg_test["target"], reg_test["y_pred"], zero_division=0)
    r_reg = recall_score(reg_test["target"], reg_test["y_pred"], zero_division=0)
    
    f1_irreg = f1_score(irreg_test["target"], irreg_test["y_pred"], zero_division=0)
    p_irreg = precision_score(irreg_test["target"], irreg_test["y_pred"], zero_division=0)
    r_irreg = recall_score(irreg_test["target"], irreg_test["y_pred"], zero_division=0)
    
    drop = f1_reg - f1_irreg
    
    return {
        "regular_f1": f1_reg,
        "regular_precision": p_reg,
        "regular_recall": r_reg,
        "irregular_f1": f1_irreg,
        "irregular_precision": p_irreg,
        "irregular_recall": r_irreg,
        "f1_drop": drop,
    }


def predict_store_nudges(
    store_id: str,
    df_history: pd.DataFrame,
    clf: HistGradientBoostingClassifier,
    catalog_products: Dict[str, Dict[str, Any]],
    current_order_skus: List[str],
    threshold: float = 0.7,
    max_nudges: int = 2,
    declined_skus: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Predicts next-week (week 27) reorder nudges for a store.
    
    Reminds ONLY for SKUs missing from current message, with probability >= threshold,
    at most max_nudges, sorted by highest probability.
    Does NOT repeat any SKU in declined_skus.
    """
    declined = set(declined_skus or [])
    ordered_current = set(current_order_skus)
    
    # Store size
    st_rows = df_history[df_history["store_id"] == store_id]
    if st_rows.empty:
        return []
    store_size = st_rows["store_size"].iloc[0]
    st_size_enc = STORE_SIZE_ENCODING.get(store_size, 1)
    
    # Store's historical SKUs
    store_skus = st_rows["sku_id"].unique()
    orders_set = set(zip(st_rows["store_id"], st_rows["sku_id"], st_rows["week"]))
    
    candidates = []
    target_week = 27
    is_festival_week = 0  # week 27 is standard
    
    for sku in store_skus:
        if sku in ordered_current or sku in declined:
            continue
            
        past_weeks = [w for w in range(1, 27) if (store_id, sku, w) in orders_set]
        if not past_weeks:
            continue
            
        order_freq = len(past_weeks) / 26.0
        weeks_since = target_week - max(past_weeks)
        
        # Predict probability
        features = pd.DataFrame([{
            "order_freq": order_freq,
            "weeks_since": weeks_since,
            "is_festival_week": is_festival_week,
            "store_size": st_size_enc,
        }])
        
        proba = float(clf.predict_proba(features)[0, 1])
        if proba >= threshold:
            prod_info = catalog_products.get(sku, {})
            candidates.append({
                "sku_id": sku,
                "name": prod_info.get("name", sku),
                "brand": prod_info.get("brand", ""),
                "pack_size": prod_info.get("pack_size", ""),
                "unit_price": prod_info.get("unit_price", 0),
                "case_size": prod_info.get("case_size", 1),
                "probability": proba,
            })
            
    candidates.sort(key=lambda x: x["probability"], reverse=True)
    return candidates[:max_nudges]


if __name__ == "__main__":
    print("=== Kirana Ordering Assistant - Reorder Model Training & Evaluation ===")
    df = load_order_history()
    print(f"Loaded order history: {len(df)} records across {df['store_id'].nunique()} stores.")
    
    print("Building features (weeks 5-26, no leakage)...")
    feat_df = build_feature_dataset(df)
    
    print("Training HistGradientBoostingClassifier on weeks 5-20...")
    model, train_df, test_df = train_model(feat_df)
    
    print("\n--- Model vs Baselines on unseen weeks 21-26 ---")
    eval_table = evaluate_baselines_and_model(model, test_df)
    print(eval_table.to_string(index=False))
    
    print("\n--- Threshold Decision Table ---")
    thresh_table = compute_threshold_table(model, test_df, current_threshold=0.7)
    print(thresh_table.to_string(index=False))
    
    print("\n--- Feature Importance ---")
    imp_df = compute_feature_importance(model, test_df)
    print(imp_df.to_string(index=False))
    
    print("\n--- Synthetic-Data Bias Stress Test ---")
    stress = run_stress_test(model, test_df)
    print(f"Regular stores (ST001-ST270) F1: {stress['regular_f1']:.3f}")
    print(f"Irregular stores (ST271-ST300) F1: {stress['irregular_f1']:.3f}")
    print(f"F1 Drop under behavioral chaos: {stress['f1_drop'] * 100:.1f}%")
