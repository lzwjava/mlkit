"""RAG — Retrieval-Augmented Generation, from indexing to answering.

Implements the retrieval stack mentioned in the reference table:

* vector search structures: **Flat** (brute force), **IVF** (inverted file
  with a K-Means coarse quantiser) and **HNSW** (hierarchical navigable
  small-world graph, simplified neighbour selection)
* embedders: hashed TF-IDF (fast, no model) or a trained LM's embedding
  table (contextual)
* chunking, top-k retrieval, and answer assembly (extractive, or
  prompt-stuffed generation with a TinyGPT).
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from ..ml.cluster import KMeans


def _l2normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(n, 1e-9)


def cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Rows of A vs rows of B (both assumed normalised)."""
    return np.asarray(A, dtype=np.float32) @ np.asarray(B, dtype=np.float32).T


# --------------------------------------------------------------------------
# embedders
# --------------------------------------------------------------------------

def hash_tfidf_embed(texts: list[str], dim: int = 256, seed: int = 0,
                     ngram: int = 2) -> np.ndarray:
    """Signed feature hashing of word unigrams+bigrams with TF-IDF weights."""
    rng = np.random.default_rng(seed)
    sign = rng.choice([-1.0, 1.0], size=(dim,))
    tok_lists = [_tokens(t) for t in texts]
    df = Counter()
    for toks in tok_lists:
        df.update(set(toks))
    n_docs = max(len(texts), 1)
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, toks in enumerate(tok_lists):
        tf = Counter(toks)
        for tok, c in tf.items():
            idf = math.log((1 + n_docs) / (1 + df[tok])) + 1.0
            h = hash(tok) % dim
            out[i, h] += sign[h] * (1.0 + math.log(c)) * idf
    return _l2normalize(out)


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    grams = list(words)
    for a, b in zip(words[:-1], words[1:]):
        grams.append(a + "_" + b)
    return grams


def lm_embed(model, tokenizer, texts: list[str], batch: int = 16) -> np.ndarray:
    """Mean of the model's token embeddings per document."""
    emb_table = model.tok_emb.weight.data
    out = np.zeros((len(texts), emb_table.shape[1]), dtype=np.float32)
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        lens = []
        mats = []
        for t in chunk:
            ids = tokenizer.encode(t)[:model.block_size]
            ids = ids or [tokenizer.unk_id]
            mats.append(emb_table[ids])
            lens.append(len(ids))
        maxlen = max(lens)
        stack = np.zeros((len(mats), maxlen, emb_table.shape[1]), np.float32)
        for j, m in enumerate(mats):
            stack[j, :len(m)] = m
        mask = np.zeros((len(mats), maxlen, 1), np.float32)
        for j, L in enumerate(lens):
            mask[j, :L] = 1.0
        out[i:i + batch] = (stack * mask).sum(1) / np.maximum(
            np.array(lens, np.float32)[:, None], 1.0)
    return _l2normalize(out)


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------

def chunk_text(text: str, chunk_words: int = 40, overlap: int = 10) -> list[str]:
    words = text.split()
    if len(words) <= chunk_words:
        return [text.strip()] if text.strip() else []
    step = max(chunk_words - overlap, 1)
    return [" ".join(words[i:i + chunk_words])
            for i in range(0, len(words), step)
            if words[i:i + chunk_words]]


# --------------------------------------------------------------------------
# indexes
# --------------------------------------------------------------------------

class FlatIndex:
    """Exact cosine search (the FAISS `IndexFlatIP` analogue)."""

    kind = "flat"

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.ids: list = []

    def add(self, vectors, ids=None):
        vectors = _l2normalize(vectors)
        self.vectors = np.vstack([self.vectors, vectors])
        self.ids.extend(list(ids) if ids is not None
                        else range(len(self.ids), len(self.ids) + len(vectors)))
        return self

    def search(self, q, k: int = 5):
        q = _l2normalize(np.atleast_2d(q))
        sims = cosine_sim(q, self.vectors)[0]
        k = min(k, len(self.ids))
        idx = np.argpartition(sims, -k)[-k:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        return [(self.ids[i], float(sims[i])) for i in idx]


class IVFIndex:
    """Inverted-file index: K-Means coarse quantiser + ``nprobe`` lists."""

    kind = "ivf"

    def __init__(self, dim: int, n_list: int = 8, nprobe: int = 3, seed: int = 0):
        self.dim = dim
        self.n_list = n_list
        self.nprobe = nprobe
        self.seed = seed
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.ids: list = []
        self.centroids: np.ndarray | None = None
        self.lists: list[np.ndarray] = []

    def add(self, vectors, ids=None, train_size: int = 64):
        vectors = _l2normalize(vectors)
        ids = list(ids) if ids is not None else list(
            range(len(self.ids), len(self.ids) + len(vectors)))
        need_train = self.centroids is None or len(self.vectors) < self.n_list
        self.vectors = np.vstack([self.vectors, vectors])
        self.ids.extend(ids)
        if need_train and len(self.vectors) >= self.n_list:
            km = KMeans(k=self.n_list, seed=self.seed, n_init=3)
            km.fit(self.vectors)
            self.centroids = km.cluster_centers_
        if self.centroids is not None:
            self._assign()
        return self

    def _assign(self):
        # assign each vector to its nearest centroid (cosine)
        sims = cosine_sim(self.vectors, self.centroids)
        labels = sims.argmax(1)
        self.lists = [np.where(labels == c)[0] for c in range(self.n_list)]

    def search(self, q, k: int = 5):
        if self.centroids is None or not len(self.ids):
            return FlatIndex(self.dim).add(self.vectors, self.ids).search(q, k)
        q = _l2normalize(np.atleast_2d(q))
        cent = cosine_sim(q, self.centroids)[0]
        probes = np.argsort(cent)[::-1][:self.nprobe]
        cand = np.concatenate([self.lists[p] for p in probes]) \
            if len(probes) else np.array([], dtype=int)
        if cand.size == 0:
            return []
        sims = cosine_sim(q, self.vectors[cand])[0]
        k = min(k, cand.size)
        top = np.argsort(sims)[::-1][:k]
        return [(self.ids[int(cand[i])], float(sims[i])) for i in top]


class HNSWIndex:
    """Hierarchical NSW graph (simplified neighbour heuristic).

    Multi-layer graph: node levels follow ``-ln(U) * mL``; layer 0 contains
    all nodes. Insertion greedily descends from the entry point, then connects
    each level to its M nearest neighbours; search greedily descends and runs
    a beam (efSearch) on layer 0.
    """

    kind = "hnsw"

    def __init__(self, dim: int, M: int = 8, ef_construction: int = 32,
                 ef_search: int = 32, seed: int = 0):
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.rng = np.random.default_rng(seed)
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.ids: list = []
        self.levels: list[int] = []
        self.graphs: dict[int, dict[int, set[int]]] = {}   # level -> {node: nbrs}
        self.entry: int | None = None
        self._mL = 1.0 / math.log(max(M, 2))

    # ------------------------------------------------------------------
    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        # cosine distance for normalised vectors
        return float(1.0 - a @ b)

    def _neighbors_at(self, node: int, level: int) -> set[int]:
        return self.graphs.get(level, {}).get(node, set())

    def add(self, vectors, ids=None):
        vectors = _l2normalize(vectors)
        for i, vec in enumerate(vectors):
            node = len(self.ids)
            self.vectors = np.vstack([self.vectors, vec[None]])
            self.ids.append(ids[i] if ids is not None else node)
            level = int(-math.log(max(self.rng.random(), 1e-9)) * self._mL)
            self.levels.append(level)
            for lv in range(level + 1):
                self.graphs.setdefault(lv, {}).setdefault(node, set())
            self._insert(node, level)
        return self

    def _insert(self, node: int, level: int):
        if self.entry is None:
            self.entry = node
            return
        cur = self.entry
        # greedy descent through layers above the node's level
        for lv in range(max(self.levels[self.entry], level), level, -1):
            cur = self._greedy(cur, self.vectors[node], lv)
        # connect on each layer down to 0
        for lv in range(min(level, max(self.levels[self.entry], level)), -1, -1):
            candidates = self._search_layer(self.vectors[node], cur,
                                            self.ef_construction, lv)
            candidates = [c for c in candidates if c != node][:self.M]
            for c in candidates:
                self.graphs.setdefault(lv, {}).setdefault(node, set()).add(c)
                nb = self.graphs.setdefault(lv, {}).setdefault(c, set())
                nb.add(node)
                if len(nb) > 2 * self.M:                # prune to 2M
                    keep = sorted(nb, key=lambda x: self._dist(
                        self.vectors[x], self.vectors[c]))[:2 * self.M]
                    self.graphs[lv][c] = set(keep)
            if candidates:
                cur = candidates[0]
        if level > self.levels[self.entry]:
            self.entry = node

    def _greedy(self, start: int, vec: np.ndarray, level: int) -> int:
        cur, cur_d = start, self._dist(vec, self.vectors[start])
        changed = True
        while changed:
            changed = False
            for nb in self._neighbors_at(cur, level):
                d = self._dist(vec, self.vectors[nb])
                if d < cur_d:
                    cur, cur_d, changed = nb, d, True
        return cur

    def _search_layer(self, vec: np.ndarray, entry: int, ef: int,
                      level: int) -> list[int]:
        visited = {entry}
        d0 = self._dist(vec, self.vectors[entry])
        candidates = [(d0, entry)]
        best = [(d0, entry)]
        while candidates:
            candidates.sort(key=lambda t: t[0])
            d_cur, cur = candidates.pop(0)
            worst = max(b[0] for b in best)
            if d_cur > worst and len(best) >= ef:
                break
            for nb in self._neighbors_at(cur, level):
                if nb in visited:
                    continue
                visited.add(nb)
                d = self._dist(vec, self.vectors[nb])
                if len(best) < ef or d < max(b[0] for b in best):
                    candidates.append((d, nb))
                    best.append((d, nb))
                    best.sort(key=lambda t: t[0])
                    best = best[:ef]
        return [n for _, n in sorted(best)]

    def search(self, q, k: int = 5):
        if self.entry is None:
            return []
        q = _l2normalize(np.atleast_2d(q))[0]
        cur = self.entry
        for lv in range(max(self.levels), 0, -1):
            cur = self._greedy(cur, q, lv)
        cand = self._search_layer(q, cur, max(self.ef_search, k), 0)
        scored = sorted(((c, 1.0 - self._dist(q, self.vectors[c]))
                         for c in cand), key=lambda t: -t[1])[:k]
        return [(self.ids[c], float(s)) for c, s in scored]


INDEXES = {"flat": FlatIndex, "ivf": IVFIndex, "hnsw": HNSWIndex}


def build_index(kind: str, dim: int, vectors, ids=None, **kw):
    idx = INDEXES[kind](dim, **kw)
    idx.add(vectors, ids)
    return idx


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

class RAG:
    def __init__(self, index_kind: str = "flat", embedder: str = "tfidf",
                 dim: int = 256, chunk_words: int = 40, overlap: int = 10,
                 model=None, tokenizer=None, **kw):
        self.index_kind = index_kind
        self.embedder = embedder
        self.dim = dim
        self.chunk_words = chunk_words
        self.overlap = overlap
        self.model = model
        self.tokenizer = tokenizer
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.index = None
        self._kw = kw

    # ------------------------------------------------------------------
    def _embed(self, texts: list[str]) -> np.ndarray:
        if self.embedder == "lm":
            if self.model is None or self.tokenizer is None:
                raise ValueError("lm embedder needs a model + tokenizer")
            return lm_embed(self.model, self.tokenizer, texts)
        return hash_tfidf_embed(texts, dim=self.dim)

    def index_documents(self, docs: list[tuple[str, str]], verbose=True):
        """docs: list of (source_name, text)."""
        all_chunks, all_src = [], []
        for name, text in docs:
            for c in chunk_text(text, self.chunk_words, self.overlap):
                all_chunks.append(c)
                all_src.append(name)
        if not all_chunks:
            raise ValueError("no chunks produced from documents")
        vecs = self._embed(all_chunks)
        dim = vecs.shape[1]
        self.index = build_index(self.index_kind, dim, vecs,
                                 ids=list(range(len(all_chunks))), **self._kw)
        self.chunks = all_chunks
        self.sources = all_src
        if verbose:
            print(f"    indexed {len(all_chunks)} chunks from "
                  f"{len({s for s in all_src})} documents "
                  f"(dim={dim}, index={self.index_kind}, "
                  f"embedder={self.embedder})")
        return self

    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        vec = self._embed([query])
        hits = self.index.search(vec, k=k)
        return [{"rank": i + 1, "score": s, "source": self.sources[j],
                 "text": self.chunks[j]}
                for i, (j, s) in enumerate(hits)]

    def answer_extractive(self, query: str, k: int = 2) -> dict:
        hits = self.retrieve(query, k=k)
        joined = " ".join(h["text"] for h in hits)
        return {"mode": "extractive", "hits": hits, "answer": joined.strip()}

    def answer_generate(self, query: str, k: int = 3,
                        max_new_tokens: int = 60, seed: int = 0) -> dict:
        """Prompt-stuffed generation: retrieve → condition → continue."""
        from .generate import generate
        hits = self.retrieve(query, k=k)
        context = "\n".join(f"- {h['text']}" for h in hits)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        ids = self.tokenizer.encode(prompt)[-self.model.block_size + 40:]
        out = generate(self.model, ids, max_new_tokens=max_new_tokens,
                       temperature=0.8, top_k=8, seed=seed)
        completion = self.tokenizer.decode(out[len(ids):])
        return {"mode": "generative", "hits": hits, "prompt": prompt,
                "answer": completion.strip()}