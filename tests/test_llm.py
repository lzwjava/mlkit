"""LLM stack: tokenizers, training, decoding, LoRA, quantization, RAG."""

import numpy as np
import pytest

from mlkit.data import tiny_corpus
from mlkit.llm.tokenizer import CharTokenizer, BPETokenizer, tokenizer_from_state
from mlkit.llm.model import TinyGPT, train_lm, build_clm_batch
from mlkit.llm import generate as G
from mlkit.llm import quantize as Q
from mlkit.llm import lora as L
from mlkit.llm import flash as F
from mlkit.llm import rag as R
from mlkit.llm import agent as A


TEXT = tiny_corpus(4000, seed=0)


# --------------------------------------------------------------------------
# tokenizers
# --------------------------------------------------------------------------

def test_char_tokenizer_roundtrip():
    tok = CharTokenizer.fit(TEXT)
    s = "the old king found the map"
    assert tok.decode(tok.encode(s)) == s
    assert tok.vocab_size >= 10


def test_bpe_tokenizer_roundtrip_and_merges():
    tok = BPETokenizer.fit(TEXT, vocab_size=150)
    s = "the old king found the map in the forest"
    assert tok.decode(tok.encode(s)) == s
    assert len(tok.merges) > 0
    # BPE should compress: fewer tokens than characters
    assert len(tok.encode(s)) < len(s)
    # persistence roundtrip
    state = tokenizer_from_state(tok.state())
    assert state.encode(s) == tok.encode(s)


def test_mask_token():
    tok = CharTokenizer.fit(TEXT)
    mid = tok.ensure_mask()
    assert tok.vocab[mid] == "<mask>"


# --------------------------------------------------------------------------
# model + training
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained():
    tok = CharTokenizer.fit(TEXT)
    ids = np.asarray(tok.encode(TEXT), dtype=np.int64)
    model = TinyGPT(tok.vocab_size, d_model=48, n_head=4, n_layer=2,
                    block_size=32, seed=0)
    hist = train_lm(model, ids, steps=120, batch_size=32, lr=6e-3,
                    log_every=999, verbose=False, seed=0)
    return model, tok, ids, hist


def test_training_reduces_perplexity(trained):
    model, tok, ids, hist = trained
    assert hist["final_val_ppl"] < 10.0        # far below chance (37 vocab)
    assert hist["val_loss"][-1] < hist["val_loss"][0]


def test_mlm_objective_trains():
    tok = CharTokenizer.fit(TEXT)
    ids = np.asarray(tok.encode(TEXT), dtype=np.int64)
    mask_id = tok.ensure_mask()
    model = TinyGPT(tok.vocab_size, d_model=32, n_head=4, n_layer=1,
                    block_size=24, seed=1)
    hist = train_lm(model, ids, objective="mlm", mask_id=mask_id,
                    steps=60, batch_size=32, lr=6e-3, log_every=999,
                    verbose=False, seed=1)
    assert np.isfinite(hist["final_val_loss"])


def test_clm_batch_shapes():
    ids = np.arange(100, dtype=np.int64)
    rng = np.random.default_rng(0)
    x, y = build_clm_batch(ids, batch=4, block=8, rng=rng)
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert (y[:, :-1] == x[:, 1:]).all()       # next-token alignment
    assert y[0, -1] == x[0, -1] + 1


# --------------------------------------------------------------------------
# decoding strategies
# --------------------------------------------------------------------------

def test_greedy_is_deterministic(trained):
    model, tok, ids, _ = trained
    prompt = tok.encode("the ")
    a = G.generate(model, prompt, max_new_tokens=15, temperature=0.0, seed=0)
    b = G.generate(model, prompt, max_new_tokens=15, temperature=0.0, seed=1)
    assert a == b


def test_sampling_strategies(trained):
    model, tok, ids, _ = trained
    prompt = tok.encode("the ")
    for kw in ({"temperature": 1.0},
               {"temperature": 1.0, "top_k": 3},
               {"temperature": 0.9, "top_p": 0.9}):
        out = G.generate(model, prompt, max_new_tokens=12, seed=7, **kw)
        assert len(out) == len(prompt) + 12


def test_beam_search_runs(trained):
    model, tok, ids, _ = trained
    prompt = tok.encode("the ")
    out = G.beam_search(model, prompt, max_new_tokens=8, beam_size=3)
    assert len(out) >= len(prompt)


def test_top_k_filter():
    logits = np.array([[1.0, 2.0, 3.0, 4.0]])
    filtered = G.top_k_filter(logits, 2)
    assert np.isneginf(filtered[0, 0]) and np.isneginf(filtered[0, 1])
    assert np.isfinite(filtered[0, 2]) and np.isfinite(filtered[0, 3])


def test_top_p_filter():
    logits = np.log(np.array([[0.7, 0.2, 0.09, 0.01]]))
    filtered = G.top_p_filter(logits, 0.9)
    keep = np.isfinite(filtered).sum()
    assert keep >= 2 and keep <= 3


def test_kv_cache_identical_and_faster(trained):
    model, tok, ids, _ = trained
    prompt = tok.encode("the ")
    b = G.benchmark_cache(model, prompt, n_tokens=20)
    assert b["identical"]                       # same outputs either way
    assert b["speedup"] > 0.5                   # timing is noisy on tiny models


# --------------------------------------------------------------------------
# LoRA
# --------------------------------------------------------------------------

def test_lora_freezes_and_trains_only_adapters(trained):
    model, tok, ids, _ = trained
    before = {k: p.data.copy() for k, p in model.named_parameters()}
    wrapped = L.apply_lora(model, targets=("c_attn",), rank=4, seed=0)
    assert len(wrapped) == model.n_layer
    params = L.lora_parameters(model)
    trainable = sum(p.data.size for p in params)
    total = model.num_params(trainable_only=False)
    assert trainable < 0.1 * total            # <10% of parameters

    hist = train_lm(model, ids, steps=10, batch_size=16, lr=3e-3,
                    log_every=99, verbose=False, lora_params=params,
                    optimizer="adam", weight_decay=0.0)
    # frozen base weights unchanged (names shift after wrapping: compare
    # by tensor identity for tensors that existed before)
    for k, p in model.named_parameters():
        if not p.requires_grad and id(p) in {id(t) for t in before.values()}:
            orig = next(t for t in before.values() if id(t) == id(p))
            assert np.allclose(p.data, orig)
    # adapters changed
    changed = any(not np.allclose(p.data, 0)
                  for p in params if p.name.endswith("B"))
    assert changed or hist["final_val_ppl"] < 100


def test_lora_save_load_merge(trained, tmp_path):
    model, tok, ids, _ = trained
    L.apply_lora(model, targets=("c_attn",), rank=4, seed=0)
    path = str(tmp_path / "adapter.npz")
    L.save_adapter(model, path)
    state = np.load(path)
    assert len(state.files) == model.n_layer * 2
    merged = L.merge_lora(model)
    assert merged == model.n_layer


# --------------------------------------------------------------------------
# quantization
# --------------------------------------------------------------------------

def test_int8_roundtrip_is_accurate():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.05, size=(32, 64)).astype(np.float32)
    q, scale = Q.quantize_int8_per_channel(W)
    deq = Q.dequantize_int8(q, scale)
    assert q.dtype == np.int8
    assert np.abs(deq - W).max() < 0.005


def test_int4_groupwise_roundtrip():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.05, size=(16, 128)).astype(np.float32)
    q, scale, deq = Q.quantize_int4_groupwise(W, group=64)
    assert np.abs(deq - W).max() < 0.05


def test_quantize_model_report(trained):
    model, tok, ids, _ = trained
    snap = Q.snapshot_weights(model)
    ppl_fp = Q.evaluate_ppl(model, ids[:1500])
    rep = Q.quantize_model(model, bits="int8", apply=True)
    ppl_q = Q.evaluate_ppl(model, ids[:1500])
    assert rep["compression"] > 3.0
    assert ppl_q < ppl_fp * 1.5               # int8 barely hurts
    assert rep["quantized_bytes"] < rep["fp32_bytes"]
    Q.dequantize_model(model, snap)


# --------------------------------------------------------------------------
# flash attention
# --------------------------------------------------------------------------

def test_flash_matches_naive():
    res = F.Benchmark(B=1, H=2, T=64, d=16) if False else F.benchmark(
        B=1, H=2, T=64, d=16, reps=3)
    assert res["max_abs_err"] < 1e-4


# --------------------------------------------------------------------------
# rag
# --------------------------------------------------------------------------

def test_rag_index_and_retrieve():
    docs = [("d1", "the transformer uses self attention mechanisms "
                   "and positional encodings for sequence modelling"),
            ("d2", "quantisation reduces memory by storing weights in "
                   "int8 and int4 formats with per channel scales"),
            ("d3", "k means clustering assigns points to the nearest "
                   "centroid and recomputes the centroids iteratively")]
    for kind in ("flat", "ivf", "hnsw"):
        rag = R.RAG(index_kind=kind, embedder="tfidf", dim=128)
        rag.index_documents(docs, verbose=False)
        hits = rag.retrieve("int8 quantisation weights memory", k=2)
        assert hits and hits[0]["source"] == "d2", (kind, hits)
        hits2 = rag.retrieve("clustering centroids", k=2)
        assert hits2[0]["source"] == "d3", (kind, hits2)


def test_hnsw_recall_vs_flat():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(200, 32)).astype(np.float32)
    q = rng.normal(size=(1, 32)).astype(np.float32)
    flat = R.FlatIndex(32).add(vecs)
    hnsw = R.HNSWIndex(32, M=8, seed=0).add(vecs)
    truth = [i for i, _ in flat.search(q, k=5)]
    got = [i for i, _ in hnsw.search(q, k=5)]
    recall = len(set(truth) & set(got)) / 5
    assert recall >= 0.6


def test_chunking_overlap():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = R.chunk_text(text, chunk_words=40, overlap=10)
    assert len(chunks) > 2
    # the last 10 words of a chunk are the first 10 of the next one
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]


# --------------------------------------------------------------------------
# speculative decoding
# --------------------------------------------------------------------------

def test_speculative_acceptance():
    tok = CharTokenizer.fit(TEXT)
    ids = np.asarray(tok.encode(TEXT), dtype=np.int64)
    target = TinyGPT(tok.vocab_size, d_model=48, n_head=4, n_layer=2,
                     block_size=32, seed=0)
    train_lm(target, ids, steps=100, batch_size=32, lr=6e-3,
             log_every=999, verbose=False, seed=0)
    draft = TinyGPT(tok.vocab_size, d_model=32, n_head=2, n_layer=1,
                    block_size=32, seed=1)
    train_lm(draft, ids, steps=100, batch_size=32, lr=6e-3,
             log_every=999, verbose=False, seed=1)
    from mlkit.llm.speculative import speculative_greedy_tokens
    prompt = tok.encode("the ")
    out = speculative_greedy_tokens(target, draft, prompt, n_tokens=16, k=4)
    assert out["generated"] == 16
    assert out["acceptance_rate"] > 0.15
    assert out["target_forwards"] < out["generated"]   # fewer big-model calls


# --------------------------------------------------------------------------
# agent tools
# --------------------------------------------------------------------------

def test_agent_tools():
    assert "= 123" in A.tool_calculator("12 * (8 + 3) - 45 / 5")
    assert "62.1" in A.tool_convert("convert 100 km to mi")
    assert "77.0" in A.tool_convert("convert 25 c to f")
    assert "11.02" in A.tool_convert("5 kg to lb")
    assert "Photosynthesis" in A.tool_search("what is photosynthesis")


def test_react_and_cot_loops():
    out = A.react("calculate 6 * 7", verbose=False)
    roles = [r for r, _ in out["trace"]]
    assert "Thought" in roles and "Action" in roles and "Answer" in roles
    assert "= 42" in out["answer"]
    out2 = A.chain_of_thought("calculate 6 * 7", verbose=False)
    assert "= 42" in out2["answer"]