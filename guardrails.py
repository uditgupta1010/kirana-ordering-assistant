"""guardrails.py - Deterministic guardrails, rule parsers, catalog math, and sanity checks.

Core principle: The LLM proposes, Python validates.
Product IDs, prices, and totals always come from the catalog, never from the LLM.
"""

from __future__ import annotations
import os
import json
import yaml
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


def load_ordering_rules(filepath: str = "ordering-rules.md") -> Tuple[Dict[str, Any], str]:
    """Reads ordering-rules.md fresh from disk on every run.
    
    Extracts the YAML front-matter into a dictionary of settings,
    and returns the remaining markdown body as the system prompt instructions.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing required rules file: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by front-matter delimiter '---'
    parts = content.split("---")
    if len(parts) >= 3:
        front_matter_raw = parts[1]
        rules_text = "---".join(parts[2:]).strip()
        settings = yaml.safe_load(front_matter_raw) or {}
    else:
        settings = {}
        rules_text = content.strip()
    
    # Sensible defaults matching the spec
    defaults = {
        "match_threshold_high": 85,
        "match_threshold_low": 55,
        "match_margin": 8,
        "nudge_probability_threshold": 0.7,
        "max_nudges_per_order": 2,
        "quantity_flag_multiple": 3,
        "retrieval_top_k": 5,
    }
    for k, v in defaults.items():
        if k not in settings:
            settings[k] = v
            
    return settings, rules_text


def load_message_templates(filepath: str = "message-templates.md") -> Dict[str, str]:
    """Reads message-templates.md fresh from disk on every run.
    
    Parses sections delimited by '## key' headings.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing required templates file: {filepath}")
        
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    templates: Dict[str, str] = {}
    current_key: Optional[str] = None
    current_lines: List[str] = []
    
    for line in content.splitlines():
        if line.startswith("## "):
            if current_key:
                templates[current_key] = "\n".join(current_lines).strip()
            current_key = line[3:].strip()
            current_lines = []
        elif current_key:
            current_lines.append(line)
            
    if current_key:
        templates[current_key] = "\n".join(current_lines).strip()
        
    return templates


def render_template(templates: Dict[str, str], key: str, **kwargs: Any) -> str:
    """Renders a named template with given keyword arguments, safely handling missing keys."""
    raw = templates.get(key, "")
    if not raw:
        return f"[{key}]"
    try:
        return raw.format(**kwargs)
    except KeyError:
        # Fallback if a placeholder is missing
        res = raw
        for k, v in kwargs.items():
            res = res.replace(f"{{{k}}}", str(v))
        return res


def load_catalog(filepath: str = "data/catalog.json") -> Dict[str, Any]:
    """Loads and validates catalog.json. Builds lookup tables."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing catalog file: {filepath}")
        
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    products = data.get("products", [])
    sku_to_product: Dict[str, Dict[str, Any]] = {p["sku_id"]: p for p in products}
    data["sku_to_product"] = sku_to_product
    return data


def compute_usual_quantities(order_history_df: pd.DataFrame) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    """Computes median quantity ordered (in pieces) per (store_id, sku_id),
    and catalog-wide median per sku_id.
    """
    if order_history_df is None or order_history_df.empty:
        return {}, {}
        
    store_sku_medians = (
        order_history_df.groupby(["store_id", "sku_id"])["qty"]
        .median()
        .to_dict()
    )
    catalog_medians = (
        order_history_df.groupby("sku_id")["qty"]
        .median()
        .to_dict()
    )
    return store_sku_medians, catalog_medians


def check_quantity_flag(
    store_id: str,
    sku_id: str,
    total_pieces: int,
    store_sku_medians: Dict[Tuple[str, str], float],
    catalog_medians: Dict[str, float],
    flag_multiple: float = 3.0,
) -> Tuple[bool, str, float]:
    """Flags any line where total_pieces > flag_multiple * usual_quantity.
    
    Usual quantity is the store's historical median for that SKU,
    falling back to the catalog-wide median for that SKU if the store hasn't ordered it.
    """
    usual = store_sku_medians.get((store_id, sku_id))
    is_store_specific = True
    if usual is None or usual <= 0:
        usual = catalog_medians.get(sku_id, 24.0)
        is_store_specific = False
        
    threshold = usual * flag_multiple
    if total_pieces > threshold:
        origin = "store usual" if is_store_specific else "catalog average"
        reason = f"Ordered {total_pieces} pcs (>{flag_multiple:.0f}x {origin} of {usual:.0f} pcs)"
        return True, reason, usual
    return False, "", usual


def calculate_line_total(
    sku_id: str,
    qty: int,
    pieces_per_unit: int,
    sku_to_product: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Strictly computes pieces, unit price, and line total using catalog truth."""
    product = sku_to_product.get(sku_id)
    if not product:
        raise ValueError(f"SKU '{sku_id}' not found in catalog.")
        
    unit_price = float(product.get("unit_price", 0))
    pieces = int(qty) * int(pieces_per_unit)
    line_total = pieces * unit_price
    
    return {
        "sku_id": sku_id,
        "name": product["name"],
        "pack_size": product.get("pack_size", ""),
        "qty": qty,
        "pieces_per_unit": pieces_per_unit,
        "pieces": pieces,
        "unit_price": unit_price,
        "line_total": line_total,
    }


def is_prompt_injection(message: str) -> bool:
    """Heuristic check for adversarial prompt injections in customer order messages."""
    msg = message.lower()
    suspicious_patterns = [
        r"ignore (all )?(previous|above) (instructions|rules)",
        r"system\s*:",
        r"you are now (an? )?admin",
        r"give (\d+%\s*)?discount",
        r"set price",
        r"mark (this )?order as paid",
        r"free items?",
        r"skip confirmation",
        r"ye instruction hai",
        r"discount on everything",
    ]
    for pattern in suspicious_patterns:
        if re.search(pattern, msg):
            return True
    return False
