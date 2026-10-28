"""matching.py - RAG retrieval (fuzzy + char n-gram TF-IDF), candidate validation, and unit conversion.

Core principle: Python validates every match and strictly computes unit conversion.
The LLM only chooses from retrieved candidates; any other ID is rejected.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CatalogMatcher:
    """Retrieves top candidates from catalog using rapidfuzz and scikit-learn character TF-IDF."""

    def __init__(self, catalog: Dict[str, Any]):
        self.catalog = catalog
        self.products = catalog.get("products", [])
        self.sku_to_product = {p["sku_id"]: p for p in self.products}
        self.unit_map = catalog.get("unit_map", {})
        
        # Build inverted word-to-canonical unit lookup
        self.word_to_canonical: Dict[str, Tuple[str, Any]] = {}
        for canonical, udata in self.unit_map.items():
            words = udata.get("words", [])
            pieces_spec = udata.get("pieces", 1)
            for w in words:
                self.word_to_canonical[w.lower()] = (canonical, pieces_spec)
        
        # Build document texts for TF-IDF
        self.doc_texts: List[str] = []
        for p in self.products:
            parts = [p.get("name", ""), p.get("brand", ""), p.get("pack_size", "")]
            parts.extend(p.get("aliases", []))
            self.doc_texts.append(" ".join(parts).lower())
            
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

    def _extract_pack_token(self, text: str) -> Optional[str]:
        """Extracts and normalizes pack size token like 100g, 50g, 1kg, 750ml, etc."""
        m = re.search(r"\b(\d+\s*(?:g|gm|gms|kg|kilo|l|lt|ltr|litre|liter|ml))\b", text.lower())
        if m:
            t = re.sub(r"\s+", "", m.group(1))
            t = re.sub(r"gm|gms", "g", t)
            t = re.sub(r"kilo", "kg", t)
            t = re.sub(r"liter|litre|ltr|lt", "l", t)
            return t
        return None

    def retrieve_candidates(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves top-k candidates with combined score in [0, 100]."""
        q_lower = query.lower().strip()
        if not q_lower:
            return []
            
        q_vec = self.vectorizer.transform([q_lower])
        tfidf_sims = cosine_similarity(q_vec, self.tfidf_matrix)[0] # 0 to 1
        
        q_words = set(re.findall(r"\w+", q_lower))
        q_pack_token = self._extract_pack_token(q_lower)
        
        scored: List[Dict[str, Any]] = []
        for idx, p in enumerate(self.products):
            # Target strings to compare against
            targets = [p["name"].lower(), p["brand"].lower()] + [a.lower() for a in p.get("aliases", [])]
            
            best_fuzzy = 0.0
            for s in targets:
                ts_ratio = fuzz.token_sort_ratio(q_lower, s)
                r_ratio = fuzz.ratio(q_lower, s)
                set_ratio = fuzz.token_set_ratio(q_lower, s)
                
                s_words = set(re.findall(r"\w+", s))
                # Only give full token_set_ratio if there is actual word-level overlap
                if q_words.intersection(s_words):
                    f_score = max(ts_ratio, r_ratio, set_ratio)
                else:
                    f_score = max(ts_ratio, r_ratio, set_ratio * 0.70)
                    
                if f_score > best_fuzzy:
                    best_fuzzy = f_score
                    
            tfidf_score = float(tfidf_sims[idx]) * 100.0
            
            # Base combined score: 60% fuzzy, 40% character TF-IDF
            combined_score = 0.60 * best_fuzzy + 0.40 * tfidf_score
            
            # Pack size sensitivity:
            # If customer specifically typed a pack size (e.g. 100g, 50g, 1kg),
            # reward matching pack size and penalize mismatching pack size
            p_pack_token = self._extract_pack_token(p.get("pack_size", ""))
            if q_pack_token and p_pack_token:
                if q_pack_token == p_pack_token:
                    combined_score = min(100.0, combined_score + 10.0)
                else:
                    combined_score = max(0.0, combined_score - 15.0)
            
            scored.append({
                "sku_id": p["sku_id"],
                "name": p["name"],
                "pack_size": p.get("pack_size", ""),
                "unit_price": p.get("unit_price", 0),
                "case_size": p.get("case_size"),
                "box_size": p.get("box_size"),
                "strip_size": p.get("strip_size"),
                "aliases": p.get("aliases", []),
                "score": round(combined_score, 1),
                "fuzzy_score": round(best_fuzzy, 1),
                "tfidf_score": round(tfidf_score, 1),
            })
            
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def validate_and_filter_choice(
        self,
        phrase: str,
        candidates: List[Dict[str, Any]],
        llm_choice: str,
        match_threshold_high: float = 85.0,
        match_threshold_low: float = 55.0,
        match_margin: float = 8.0,
    ) -> Tuple[str, Optional[str], bool, str]:
        """Validates LLM choice against Python deterministic guardrails.
        
        Returns:
            (final_decision, chosen_sku_or_none, is_made_up_attempt, matched_via_caption)
            final_decision: "MATCH" | "ASK" | "NOT_IN_CATALOG"
        """
        if not candidates:
            return "NOT_IN_CATALOG", None, False, "No candidates found (score: 0)"
            
        top_cand = candidates[0]
        top_score = top_cand["score"]
        candidate_ids = {c["sku_id"] for c in candidates}
        
        # Guardrail 1: Forced NOT_IN_CATALOG if best retrieval score is below threshold_low
        if top_score < match_threshold_low:
            return "NOT_IN_CATALOG", None, False, f"Score {top_score} < low threshold {match_threshold_low}"
            
        # Guardrail 2: Pack-size / brand ambiguity check
        # If top 2 candidates are different pack sizes of same product/brand and scores are within match_margin
        # and the customer did not specify which one
        q_pack_token = self._extract_pack_token(phrase)
        if len(candidates) >= 2 and not q_pack_token:
            second_cand = candidates[1]
            score_diff = top_score - second_cand["score"]
            top_prod = self.sku_to_product[top_cand["sku_id"]]
            second_prod = self.sku_to_product[second_cand["sku_id"]]
            
            same_base = (
                top_prod.get("brand") == second_prod.get("brand")
                and (
                    "parle-g" in top_prod["name"].lower() and "parle-g" in second_prod["name"].lower()
                    or top_prod.get("category") == second_prod.get("category")
                )
            )
            is_generic = phrase.strip().lower() in ["parle", "parle-g", "oil", "tel", "chai", "tea", "sabun", "shampoo"]
            if (same_base and score_diff < match_margin) or is_generic:
                return "ASK", None, False, f"Forced ASK: pack size/brand ambiguity (margin {score_diff:.1f} < {match_margin})"

        # Guardrail 3: Check LLM decision
        if llm_choice == "NOT_IN_CATALOG":
            return "NOT_IN_CATALOG", None, False, f"LLM decided NOT_IN_CATALOG (top score {top_score})"
            
        if llm_choice == "ASK":
            return "ASK", None, False, f"LLM requested ASK (top score {top_score})"
            
        # Guardrail 4: Made-up product check
        # Choice must be in retrieved candidate IDs
        if llm_choice not in candidate_ids:
            # Rejection! Counted as made-up product attempt
            return "ASK", None, True, f"Rejected made-up SKU '{llm_choice}' not in candidates"
            
        # Valid choice
        chosen_cand = next(c for c in candidates if c["sku_id"] == llm_choice)
        return "MATCH", llm_choice, False, f"RAG score {chosen_cand['score']:.1f}% + LLM choice {llm_choice}"

    def convert_unit(
        self,
        qty: Optional[int],
        unit_word: Optional[str],
        sku_id: str,
    ) -> Tuple[Optional[str], Optional[int], bool, str]:
        """Maps unit word to canonical unit and computes pieces multiplier.
        
        Returns:
            (canonical_unit, pieces_per_unit, needs_followup, reason)
        """
        if qty is None or qty <= 0:
            return None, None, True, "Quantity missing or invalid"
            
        product = self.sku_to_product.get(sku_id)
        if not product:
            return None, None, True, "Product unknown"
            
        w = (unit_word or "").lower().strip()
        
        # If no unit word provided, default to piece if reasonable
        if not w:
            return "piece", 1, False, "Defaulted to piece"
            
        # Clean common Hindi suffixes or plurals
        lookup_w = w
        if lookup_w not in self.word_to_canonical:
            # Strip trailing punctuation or s
            lookup_w = re.sub(r"[^\w]", "", lookup_w)
            
        if lookup_w not in self.word_to_canonical:
            # Check special cases like kilo, kg, litre
            if lookup_w in ["kilo", "kg", "litre", "liter", "ltr"]:
                lookup_w = "piece"
            elif lookup_w.endswith("s") and lookup_w[:-1] in self.word_to_canonical:
                lookup_w = lookup_w[:-1]
                
        if lookup_w not in self.word_to_canonical:
            return None, None, True, f"Unknown unit word '{unit_word}'"
            
        canonical, pieces_spec = self.word_to_canonical[lookup_w]
        
        # Resolve pieces_spec
        if isinstance(pieces_spec, int):
            return canonical, pieces_spec, False, "OK"
            
        if isinstance(pieces_spec, str):
            # product field lookup: 'case_size', 'box_size', 'strip_size'
            val = product.get(pieces_spec)
            if val is None or val <= 0:
                # Per unit_notes: If a product has no box_size or strip_size, 'dabba' or 'ladi' is ambiguous: ask a follow-up
                return None, None, True, f"Product {product['name']} has no {pieces_spec} for unit '{canonical}'"
            return canonical, int(val), False, "OK"
            
        return "piece", 1, False, "Fallback to piece"
