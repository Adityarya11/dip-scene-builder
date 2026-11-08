"""Step 3b - cosine-similarity search. One class, same interface for every feature type.
With only ~1500 items a plain NumPy matrix product is instant, so faiss is not needed
(faiss-cpu can be dropped in: IndexFlatIP over the same normalised vectors gives identical results)."""
import numpy as np


class Index:
    def __init__(self, feats, name=""):
        self.name = name
        self.X = np.ascontiguousarray(feats, dtype=np.float32)    # (N, d), rows L2-normalised

    def search(self, q, k=10, exclude=None, mask=None):
        """Top-k most similar rows to query vector q.
        exclude: row id to skip (used for leave-one-out image queries)
        mask:    optional boolean array of allowed rows (e.g. only one category).
        Returns (ids, scores)."""
        s = self.X @ q.astype(np.float32)
        if exclude is not None:
            s[exclude] = -np.inf
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        ids = np.argsort(-s)[:k]
        return ids, s[ids]
