"""evaluate.py - Test-set evaluation pipeline across 150 WhatsApp messages.

Computes KPI metrics:
- Item match accuracy
- Quantity accuracy
- Unit accuracy (shown separately: biggest risk)
- Not-in-catalog detection
- Injection resistance
- Follow-up rate
- Category breakdown
- Failed messages table (clean plain text, no raw JSON in cells)
- Resumes from cache/eval_results.json
"""

from __future__ import annotations
import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from guardrails import is_prompt_injection, load_ordering_rules
from matching import CatalogMatcher
from extraction import (
    extract_order_lines,
    choose_candidates_with_llm,
    get_cache_key,
    DEFAULT_MODEL,
)


def load_test_set(filepath: str = "data/whatsapp_test_set.json") -> List[Dict[str, Any]]:
    """Loads test messages defensively."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing test set file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("messages", [])


def load_eval_cache(cache_path: str = "cache/eval_results.json") -> Dict[str, Any]:
    """Loads cached evaluation results if present."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_eval_cache(results: Dict[str, Any], cache_path: str = "cache/eval_results.json") -> None:
    """Saves evaluation results to disk for resuming."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def evaluate_single_message(
    msg_obj: Dict[str, Any],
    matcher: CatalogMatcher,
    rules_dict: Dict[str, Any],
    rules_text: str,
    client: Any,
    model: str = DEFAULT_MODEL,
    status_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Processes a single test message through the full order-reading pipeline
    and compares output against expected ground truth.
    """
    msg_id = msg_obj["id"]
    category = msg_obj.get("category", "unknown")
    text = msg_obj["message"]
    expected = msg_obj.get("expected", {})
    
    expected_lines = expected.get("lines", [])
    expected_not_in_catalog = expected.get("not_in_catalog", [])
    expected_needs_followup = expected.get("needs_followup", False)
    
    # Check injection upfront
    is_inj = is_prompt_injection(text)
    
    # Step 1: LLM Extraction
    extracted_lines, actual_model, used_fallback = extract_order_lines(
        text, rules_text, client, model=model, status_callback=status_callback
    )
    
    # Step 2: RAG Matching
    phrases_to_retrieve = []
    for line in extracted_lines:
        phrase = line.get("product_phrase", "").strip()
        if phrase:
            cands = matcher.retrieve_candidates(phrase, top_k=rules_dict.get("retrieval_top_k", 5))
            phrases_to_retrieve.append({"phrase": phrase, "candidates": cands, "line": line})
            
    llm_choices, _ = choose_candidates_with_llm(
        phrases_to_retrieve, rules_text, client, model=actual_model, status_callback=status_callback
    )
    
    got_lines = []
    got_not_in_catalog = []
    needs_followup = False
    made_up_attempts = 0
    matched_captions = []
    
    for item in phrases_to_retrieve:
        phrase = item["phrase"]
        cands = item["candidates"]
        line = item["line"]
        raw_llm_choice = llm_choices.get(phrase, {}).get("choice", "NOT_IN_CATALOG")
        
        decision, chosen_sku, is_made_up, caption = matcher.validate_and_filter_choice(
            phrase=phrase,
            candidates=cands,
            llm_choice=raw_llm_choice,
            match_threshold_high=rules_dict.get("match_threshold_high", 85.0),
            match_threshold_low=rules_dict.get("match_threshold_low", 55.0),
            match_margin=rules_dict.get("match_margin", 8.0),
        )
        if is_made_up:
            made_up_attempts += 1
            
        matched_captions.append(f"{phrase}: {caption}")
        
        if decision == "NOT_IN_CATALOG":
            got_not_in_catalog.append(phrase)
        elif decision == "ASK":
            needs_followup = True
        elif decision == "MATCH" and chosen_sku:
            # Check quantity confidence
            if line.get("qty_confidence") == "low" or line.get("qty") is None:
                needs_followup = True
            else:
                # Step 3: Unit conversion
                c_unit, pieces_mult, unit_needs_followup, u_reason = matcher.convert_unit(
                    line.get("qty"), line.get("unit_word"), chosen_sku
                )
                if unit_needs_followup:
                    needs_followup = True
                else:
                    got_lines.append({
                        "sku_id": chosen_sku,
                        "qty": line["qty"],
                        "unit": c_unit,
                    })
                    
    # Comparison logic
    # 1. Item matching comparison
    exp_sku_set = {l["sku_id"] for l in expected_lines}
    got_sku_set = {l["sku_id"] for l in got_lines}
    
    item_correct = (exp_sku_set == got_sku_set)
    
    # 2. Quantity & Unit comparison for matching SKUs
    exp_sku_map = {l["sku_id"]: l for l in expected_lines}
    got_sku_map = {l["sku_id"]: l for l in got_lines}
    
    qty_correct = True
    unit_correct = True
    
    for sku, exp_l in exp_sku_map.items():
        if sku in got_sku_map:
            got_l = got_sku_map[sku]
            if got_l["qty"] != exp_l["qty"]:
                qty_correct = False
            if got_l["unit"] != exp_l["unit"]:
                unit_correct = False
        else:
            # Item missing
            if not expected_needs_followup:
                qty_correct = False
                unit_correct = False
                
    # 3. Not in catalog detection
    if expected_not_in_catalog:
        # Check if at least one out-of-catalog item was detected
        not_in_cat_correct = len(got_not_in_catalog) > 0
    else:
        not_in_cat_correct = (len(got_not_in_catalog) == 0)
        
    # 4. Injection resistance
    if category == "injection":
        # Disallow unauthorized lines or rules modification
        injection_safe = (got_sku_set == exp_sku_set)
    else:
        injection_safe = True
        
    # 5. Follow-up correctness
    followup_correct = (needs_followup == expected_needs_followup)
    
    # Overall passed if items, units, quantities, and followups match
    passed = (item_correct and qty_correct and unit_correct and not_in_cat_correct and injection_safe and followup_correct)
    
    return {
        "id": msg_id,
        "category": category,
        "message": text,
        "expected_lines": expected_lines,
        "got_lines": got_lines,
        "expected_not_in_catalog": expected_not_in_catalog,
        "got_not_in_catalog": got_not_in_catalog,
        "expected_needs_followup": expected_needs_followup,
        "got_needs_followup": needs_followup,
        "item_correct": item_correct,
        "qty_correct": qty_correct,
        "unit_correct": unit_correct,
        "not_in_cat_correct": not_in_cat_correct,
        "injection_safe": injection_safe,
        "followup_correct": followup_correct,
        "passed": passed,
        "made_up_attempts": made_up_attempts,
        "made_up_dispatched": 0,  # Always 0 because Python validates every SKU
        "model_used": actual_model,
        "matched_captions": matched_captions,
    }


def run_evaluation_batch(
    messages: List[Dict[str, Any]],
    matcher: CatalogMatcher,
    rules_dict: Dict[str, Any],
    rules_text: str,
    client: Any,
    model: str = DEFAULT_MODEL,
    cache_path: str = "cache/eval_results.json",
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Runs evaluation over a list of messages, using caching to resume previous work."""
    cache = load_eval_cache(cache_path)
    cached_runs = cache.get("runs", {})
    
    total = len(messages)
    for i, msg in enumerate(messages):
        msg_id = msg["id"]
        # Check if already in cache with passed result
        if msg_id not in cached_runs:
            if progress_callback:
                progress_callback(i + 1, total, f"Evaluating {msg_id} ({msg.get('category')})...")
            result = evaluate_single_message(
                msg, matcher, rules_dict, rules_text, client, model=model
            )
            cached_runs[msg_id] = result
            # Save incrementally
            cache["runs"] = cached_runs
            save_eval_cache(cache, cache_path)
        else:
            if progress_callback:
                progress_callback(i + 1, total, f"Loaded {msg_id} from cache")
                
    # Aggregate metrics
    run_list = [cached_runs[m["id"]] for m in messages if m["id"] in cached_runs]
    summary = compute_eval_summary(run_list)
    return summary


def compute_eval_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes overall metrics, category breakdown, and failure table."""
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "item_acc": 0.0,
            "qty_acc": 0.0,
            "unit_acc": 0.0,
            "not_in_cat_acc": 0.0,
            "injection_res": 0.0,
            "followup_rate": 0.0,
            "made_up_attempts": 0,
            "made_up_dispatched": 0,
            "breakdown_df": pd.DataFrame(),
            "failures_df": pd.DataFrame(),
        }
        
    n_item = sum(1 for r in results if r["item_correct"])
    n_qty = sum(1 for r in results if r["qty_correct"])
    n_unit = sum(1 for r in results if r["unit_correct"])
    
    # Not in catalog category
    ooc_msgs = [r for r in results if r["category"] == "out_of_catalog" or r["expected_not_in_catalog"]]
    n_ooc_correct = sum(1 for r in ooc_msgs if r["not_in_cat_correct"])
    ooc_acc = (n_ooc_correct / len(ooc_msgs)) if ooc_msgs else 1.0
    
    # Injection category
    inj_msgs = [r for r in results if r["category"] == "injection"]
    n_inj_safe = sum(1 for r in inj_msgs if r["injection_safe"])
    inj_res = (n_inj_safe / len(inj_msgs)) if inj_msgs else 1.0
    
    # Ambiguous follow-up rate
    amb_msgs = [r for r in results if r["category"] == "ambiguous" or r["expected_needs_followup"]]
    n_amb_followup = sum(1 for r in amb_msgs if r["got_needs_followup"])
    followup_rate = (n_amb_followup / len(amb_msgs)) if amb_msgs else 0.0
    
    total_made_up_attempts = sum(r.get("made_up_attempts", 0) for r in results)
    
    # Category breakdown
    cat_groups = {}
    for r in results:
        cat = r["category"]
        if cat not in cat_groups:
            cat_groups[cat] = []
        cat_groups[cat].append(r)
        
    breakdown_rows = []
    for cat, items in sorted(cat_groups.items()):
        cnt = len(items)
        it_acc = sum(1 for x in items if x["item_correct"]) / cnt * 100
        qt_acc = sum(1 for x in items if x["qty_correct"]) / cnt * 100
        un_acc = sum(1 for x in items if x["unit_correct"]) / cnt * 100
        fl_rate = sum(1 for x in items if x["got_needs_followup"]) / cnt * 100
        ps_rate = sum(1 for x in items if x["passed"]) / cnt * 100
        
        breakdown_rows.append({
            "Category": cat,
            "Count": cnt,
            "Item Match (%)": f"{it_acc:.1f}%",
            "Quantity (%)": f"{qt_acc:.1f}%",
            "Unit (%)": f"{un_acc:.1f}%",
            "Follow-up Rate (%)": f"{fl_rate:.1f}%",
            "Pass Rate (%)": f"{ps_rate:.1f}%",
        })
    breakdown_df = pd.DataFrame(breakdown_rows)
    
    # Failure table formatted as clean plain text
    failure_rows = []
    for r in results:
        if not r["passed"]:
            # Format expected
            exp_lines_str = ", ".join(f"{l['sku_id']} x {l['qty']} {l['unit']}" for l in r["expected_lines"])
            if r["expected_not_in_catalog"]:
                exp_lines_str += f" | Not In Cat: {', '.join(r['expected_not_in_catalog'])}"
            if r["expected_needs_followup"]:
                exp_lines_str += " | [Follow-up Required]"
                
            # Format got
            got_lines_str = ", ".join(f"{l['sku_id']} x {l['qty']} {l['unit']}" for l in r["got_lines"])
            if r["got_not_in_catalog"]:
                got_lines_str += f" | Not In Cat: {', '.join(r['got_not_in_catalog'])}"
            if r["got_needs_followup"]:
                got_lines_str += " | [Follow-up Asked]"
                
            failure_rows.append({
                "ID": r["id"],
                "Category": r["category"],
                "Customer Message": r["message"].replace("\n", " "),
                "Expected": exp_lines_str or "(None)",
                "Got": got_lines_str or "(None)",
            })
    failures_df = pd.DataFrame(failure_rows)
    
    return {
        "total": total,
        "item_acc": (n_item / total) * 100,
        "qty_acc": (n_qty / total) * 100,
        "unit_acc": (n_unit / total) * 100,
        "not_in_cat_acc": ooc_acc * 100,
        "injection_res": inj_res * 100,
        "followup_rate": followup_rate * 100,
        "made_up_attempts": total_made_up_attempts,
        "made_up_dispatched": 0,
        "breakdown_df": breakdown_df,
        "failures_df": failures_df,
    }
