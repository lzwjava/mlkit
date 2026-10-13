"""Tokenizers: character-level and Byte-Pair Encoding (BPE, from scratch).

Both tokenizers implement the common interface::

    tok = CharTokenizer.fit(text)          # or BPETokenizer.fit(text, vocab_size)
    ids = tok.encode("hello world")
    text = tok.decode(ids)
    tok.save("tok.json"); tok = load_tokenizer("tok.json")

Special ids: ``<pad>=0, <unk>=1, <bos>=2, <eos>=3`` for char tokenizers,
and are reserved the same way for BPE.
"""

from __future__ import annotations

import json
import re
from collections import Counter

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WS = "\u2581"          # sentencepiece-style word-start marker


# --------------------------------------------------------------------------

class CharTokenizer:
    """One id per character (including spaces and punctuation)."""

    kind = "char"

    def __init__(self, vocab: list[str]):
        self.vocab = list(vocab)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    # -- interface --------------------------------------------------------
    @classmethod
    def fit(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        # specials first, then corpus characters
        vocab = SPECIALS + [c for c in chars if c not in SPECIALS]
        return cls(vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def mask_id(self) -> int | None:
        """BPE/char tokenizers learn <mask> lazily (see `ensure_mask`)."""
        return self.stoi.get("<mask>", None)

    def ensure_mask(self) -> int:
        if "<mask>" not in self.stoi:
            self.vocab.append("<mask>")
            self.stoi["<mask>"] = len(self.vocab) - 1
            self.itos[len(self.vocab) - 1] = "<mask>"
        return self.stoi["<mask>"]

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids) -> str:
        return "".join(self.itos.get(int(i), "") for i in ids
                       if int(i) >= len(SPECIALS) or True)

    # -- persistence ------------------------------------------------------
    def save(self, path: str):
        with open(path, "w") as fh:
            json.dump({"kind": self.kind, "vocab": self.vocab}, fh)

    def state(self) -> dict:
        return {"kind": self.kind, "vocab": self.vocab}


class BPETokenizer:
    """Byte-Pair Encoding over characters (Sennrich et al. style).

    Training counts symbol-pair frequencies and merges the most frequent
    pair ``vocab_size`` times; encoding applies merges by rank.
    """

    kind = "bpe"

    def __init__(self, vocab: list[str], merges: list[list[str]]):
        self.vocab = list(vocab)
        self.stoi = {tok: i for i, tok in enumerate(vocab)}
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}

    # ------------------------------------------------------------------
    @staticmethod
    def _word_to_symbols(word: str) -> tuple[str, ...]:
        return tuple(word) if word else tuple()

    def _bpe_word(self, symbols: tuple[str, ...]) -> tuple[str, ...]:
        symbols = list(symbols)
        while len(symbols) > 1:
            pairs = list(zip(symbols[:-1], symbols[1:]))
            rank = min((self.ranks.get(p, float("inf")) for p in pairs),
                       default=float("inf"))
            if rank == float("inf"):
                break
            best = next(p for p in pairs if self.ranks.get(p) == rank)
            merged = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best:
                    merged.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
        return tuple(symbols)

    # ------------------------------------------------------------------
    @classmethod
    def fit(cls, text: str, vocab_size: int = 200, min_pair_freq: int = 2
            ) -> "BPETokenizer":
        words = Counter(_WORD_RE.findall(text.lower()))
        # vocabulary of characters (word-start marker included)
        alphabet = sorted({c for w in words for c in _WS + w})
        vocab = SPECIALS + alphabet
        merges: list[list[str]] = []

        # corpus as tuples of symbols (words prefixed with the word marker)
        corpus = {w: tuple(_WS + w) for w in words}
        target = max(vocab_size, len(vocab))
        while len(vocab) < target:
            pair_counts: Counter = Counter()
            for w, syms in corpus.items():
                if len(syms) < 2:
                    continue
                c = words[w]
                for p in zip(syms[:-1], syms[1:]):
                    pair_counts[p] += c
            if not pair_counts:
                break
            (a, b), freq = pair_counts.most_common(1)[0]
            if freq < min_pair_freq and merges:
                break
            new_tok = a + b
            merges.append([a, b])
            vocab.append(new_tok)
            # apply merge to the whole corpus
            for w, syms in list(corpus.items()):
                out, i = [], 0
                while i < len(syms):
                    if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                        out.append(new_tok)
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                corpus[w] = tuple(out)
        return cls(vocab, merges)

    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    def ensure_mask(self) -> int:
        if "<mask>" not in self.stoi:
            self.vocab.append("<mask>")
            self.stoi["<mask>"] = len(self.vocab) - 1
        return self.stoi["<mask>"]

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in _WORD_RE.findall(text.lower()):
            syms = self._bpe_word(tuple(_WS + word))
            for s in syms:
                ids.append(self.stoi.get(s, self.unk_id))
        return ids

    def decode(self, ids) -> str:
        # BPE is lossy w.r.t. exact whitespace; the word marker restores it.
        out = []
        for i in ids:
            i = int(i)
            if i in (self.pad_id, self.bos_id, self.eos_id):
                continue
            tok = self.vocab[i] if i < len(self.vocab) else ""
            out.append("?" if tok == UNK else tok)
        text = "".join(out).replace(_WS, " ")
        return re.sub(r"\s+", " ", text).strip()

    def save(self, path: str):
        with open(path, "w") as fh:
            json.dump({"kind": self.kind, "vocab": self.vocab,
                       "merges": [list(m) for m in self.merges]}, fh)

    def state(self) -> dict:
        return {"kind": self.kind, "vocab": self.vocab,
                "merges": [list(m) for m in self.merges]}


# --------------------------------------------------------------------------

def load_tokenizer(path: str):
    with open(path) as fh:
        state = json.load(fh)
    return tokenizer_from_state(state)


def tokenizer_from_state(state: dict):
    if state["kind"] == "bpe":
        return BPETokenizer(state["vocab"], state["merges"])
    return CharTokenizer(state["vocab"])


def get_tokenizer(kind: str, text: str, vocab_size: int = 200):
    if kind == "bpe":
        return BPETokenizer.fit(text, vocab_size=vocab_size)
    return CharTokenizer.fit(text)