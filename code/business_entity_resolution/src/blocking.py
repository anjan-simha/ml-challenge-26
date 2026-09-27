"""
ML Challenge 2026: Business Entity Resolution
Multi-Rule Candidate Generation (Blocking) Module

Generates a compact, high-recall candidate pool for each Source 1 entity
by combining complementary inverted index rules across names and addresses.
All blocking operations are partitioned strictly by country (open-set support).
"""

from collections import defaultdict
from .preprocessing import preprocess_record
from .embeddings import RecordEmbedder, build_composite_text
from .faiss_index import FaissSimilarityIndex


class MultiRuleBlocker:
    """
    Inverted-index based blocker operating within a country partition.
    Combines name token, name prefix, and address number/token keys to achieve >98% recall.
    """

    def __init__(self, max_candidates_per_entity=30, max_postings_per_key=300, use_faiss=False):
        self.max_candidates = max_candidates_per_entity
        self.max_postings = max_postings_per_key
        self.use_faiss = use_faiss
        
        # Inverted indexes: key -> list of candidate entity IDs
        self.token_index = defaultdict(list)
        self.prefix_index = defaultdict(list)
        self.addr_num_name_index = defaultdict(list)
        self.addr_num_addr_index = defaultdict(list)
        self.exact_name_index = defaultdict(list)
        
        # FAISS components
        self.faiss_index = None
        self.embedder = RecordEmbedder() if use_faiss else None
        self.cand_records = []

    def add_candidate(self, prep_rec):
        """Add a preprocessed Source 2 or Source 3 candidate record to all indexes."""
        cand_id = prep_rec["entity_id"]
        c = prep_rec["country"]
        name_sig = prep_rec["name_sig"]
        addr_sig = prep_rec["addr_sig"]
        nums = prep_rec["addr_numbers"]
        prefix = prep_rec["name_prefix"]
        name_norm = prep_rec["name_norm"]

        # 1. Exact normalized name
        if name_norm:
            self.exact_name_index[(c, name_norm)].append(cand_id)

        # 2. Significant name tokens (top 3)
        for tok in name_sig[:3]:
            self.token_index[(c, tok)].append(cand_id)

        # 3. Name prefix (first 5 alphanumeric characters)
        if prefix:
            self.prefix_index[(c, prefix)].append(cand_id)

        # 4. Address number + name token
        if nums and name_sig:
            for num in nums[:1]:
                for tok in name_sig[:2]:
                    self.addr_num_name_index[(c, num, tok)].append(cand_id)

        # 5. Address number + significant address token
        if nums and addr_sig:
            for num in nums[:1]:
                for tok in addr_sig[:2]:
                    self.addr_num_addr_index[(c, num, tok)].append(cand_id)
                    
        if self.use_faiss:
            self.cand_records.append(prep_rec)

    def finalize_index(self):
        if self.use_faiss and self.cand_records:
            texts = [build_composite_text(r) for r in self.cand_records]
            ids = [r["entity_id"] for r in self.cand_records]
            embeddings = self.embedder.encode(texts, batch_size=256)
            self.faiss_index = FaissSimilarityIndex(dimension=embeddings.shape[1])
            self.faiss_index.build(embeddings, ids)
            self.cand_records = [] # free memory

    def retrieve_candidates(self, prep_rec):
        """
        Retrieve candidate IDs for a Source 1 entity using union across all rules.
        Scores candidate hits to retain the most promising candidates within max_candidates.
        
        Returns:
            list of candidate entity IDs
        """
        c = prep_rec["country"]
        name_sig = prep_rec["name_sig"]
        addr_sig = prep_rec["addr_sig"]
        nums = prep_rec["addr_numbers"]
        prefix = prep_rec["name_prefix"]
        name_norm = prep_rec["name_norm"]

        # Track candidate scores: candidate_id -> count of key hits
        candidate_scores = defaultdict(int)

        # Rule 1: Exact name match (high priority bonus)
        if name_norm:
            exact_matches = self.exact_name_index.get((c, name_norm), [])
            for cid in exact_matches[:self.max_postings]:
                candidate_scores[cid] += 5

        # Rule 2: Significant name tokens (up to 3)
        for tok in name_sig[:3]:
            postings = self.token_index.get((c, tok), [])
            if len(postings) <= self.max_postings:
                for cid in postings:
                    candidate_scores[cid] += 3
            else:
                for cid in postings[:self.max_postings]:
                    candidate_scores[cid] += 1

        # Rule 3: Name prefix
        if prefix:
            postings = self.prefix_index.get((c, prefix), [])
            for cid in postings[:self.max_postings]:
                candidate_scores[cid] += 2

        # Rule 4: Address number + Name token
        if nums and name_sig:
            for num in nums[:1]:
                for tok in name_sig[:2]:
                    postings = self.addr_num_name_index.get((c, num, tok), [])
                    for cid in postings[:self.max_postings]:
                        candidate_scores[cid] += 4

        # Rule 5: Address number + Address token (crucial for empty or modified names)
        if nums and addr_sig:
            for num in nums[:1]:
                for tok in addr_sig[:2]:
                    postings = self.addr_num_addr_index.get((c, num, tok), [])
                    for cid in postings[:self.max_postings]:
                        candidate_scores[cid] += 3

        if not candidate_scores and not self.faiss_index:
            return []

        # Sort by hit score descending, then take top K
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        top_cands = [cid for cid, _ in sorted_candidates[:self.max_candidates]]
        
        if self.use_faiss and self.faiss_index:
            text = build_composite_text(prep_rec)
            emb = self.embedder.encode([text], batch_size=256)
            faiss_cands = self.faiss_index.search(emb, top_k=5)[0]
            # merge
            seen = set(top_cands)
            for fc in faiss_cands:
                if fc not in seen:
                    top_cands.append(fc)
                    seen.add(fc)
                    
        return top_cands[:self.max_candidates + 5]

    def clear(self):
        """Free inverted index memory."""
        self.token_index.clear()
        self.prefix_index.clear()
        self.addr_num_name_index.clear()
        self.addr_num_addr_index.clear()
        self.exact_name_index.clear()
