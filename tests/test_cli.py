"""CLI smoke tests: every command group runs and produces output."""

import json
import subprocess
import sys

import numpy as np
import pytest

from mlkit.cli import main


def run_cli(*argv, capsys=None):
    code = main(list(argv))
    return code, capsys.readouterr().out if capsys else ""


# --------------------------------------------------------------------------
# data / ml / bench
# --------------------------------------------------------------------------

def test_data_list(capsys):
    code, out = run_cli("data", "list", capsys=capsys)
    assert code == 0 and "moons" in out and "spirals" in out


def test_ml_classifier_json(capsys):
    code, out = run_cli("ml", "logistic-regression", "--dataset", "moons",
                        "--epochs", "80", "--json", capsys=capsys)
    res = json.loads(out)
    assert code == 0 and res["accuracy"] > 0.8


def test_ml_regression(capsys):
    code, out = run_cli("ml", "ridge", "--dataset", "regression",
                        "--json", capsys=capsys)
    res = json.loads(out)
    assert code == 0 and res["r2"] > 0.7


def test_ml_clustering(capsys):
    code, out = run_cli("ml", "kmeans", "--dataset", "gmm", "--k", "3",
                        "--json", capsys=capsys)
    res = json.loads(out)
    assert code == 0 and len(res["cluster_sizes"]) == 3


def test_ml_dimred(capsys):
    code, out = run_cli("ml", "pca", "--components", "2", "--json",
                        capsys=capsys)
    res = json.loads(out)
    assert code == 0 and abs(sum(res["explained_variance_ratio"]) - 1) < 0.05


def test_bench_runs(capsys):
    code, out = run_cli("bench", "--dataset", "moons", "--n", "200",
                        capsys=capsys)
    assert code == 0 and "xgboost" in out and "kernel-svm" in out


# --------------------------------------------------------------------------
# dl
# --------------------------------------------------------------------------

def test_dl_gradcheck_passes(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["dl", "gradcheck"])
    out = capsys.readouterr().out
    assert exc.value.code == 0
    assert "10/10 components passed" in out


def test_dl_mlp(capsys):
    code, out = run_cli("dl", "mlp", "--dataset", "moons", "--epochs", "30",
                        "--json", capsys=capsys)
    res = json.loads(out)
    assert code == 0 and res["accuracy"] > 0.85


def test_dl_optim(capsys):
    code, out = run_cli("dl", "optim", "--steps", "100", capsys=capsys)
    assert code == 0 and "rmsprop" in out


# --------------------------------------------------------------------------
# llm
# --------------------------------------------------------------------------

def test_llm_tokenize(capsys):
    code, out = run_cli("llm", "tokenize", "--tokenizer", "bpe",
                        "--text", "the old king", capsys=capsys)
    assert code == 0 and "roundtrip" in out and "the old king" in out


def test_llm_train_generate_pipeline(tmp_path, capsys):
    ckpt = str(tmp_path / "ckpt")
    code, out = run_cli("llm", "train", "--out", ckpt, "--train-steps", "60",
                        "--corpus", "4000", "--json", capsys=capsys)
    assert code == 0
    assert (tmp_path / "ckpt" / "weights.npz").exists()
    res = json.loads(out)
    assert res["val_perplexity"] < 25

    code, out = run_cli("llm", "generate", "--ckpt", ckpt, "--tokens", "20",
                        "--json", capsys=capsys)
    res = json.loads(out)
    assert code == 0 and isinstance(res["completion"], str)

    code, out = run_cli("llm", "evaluate", "--ckpt", ckpt, "--json",
                        capsys=capsys)
    res = json.loads(out)
    assert code == 0 and res["val_perplexity"] < 25

    code, out = run_cli("llm", "quantize", "--ckpt", ckpt, "--json",
                        capsys=capsys)
    res = json.loads(out)
    assert code == 0 and res["bits"][0]["compression"] > 3

    code, out = run_cli("llm", "bench-cache", "--ckpt", ckpt, "--tokens", "15",
                        "--json", capsys=capsys)
    res = json.loads(out)
    assert code == 0 and res["identical_outputs"] and res["speedup"] > 1


def test_llm_flash(capsys):
    code, out = run_cli("llm", "flash", "--seq-len", "64", "--heads", "4",
                        "--dim", "16", "--json", capsys=capsys)
    res = json.loads(out)
    assert code == 0 and res["max_abs_err"] < 1e-4


def test_llm_rag(capsys):
    code, out = run_cli("llm", "rag", "--query", "speculative decoding",
                        "--index", "flat", "--json", capsys=capsys)
    res = json.loads(out)
    assert code == 0 and "speculative" in res["hits"][0]["text"].lower()


def test_llm_agent(capsys):
    code, out = run_cli("llm", "agent", "--mode", "react",
                        "--question", "calculate 6 * 7", "--json",
                        capsys=capsys)
    res = json.loads(out)
    assert code == 0 and "42" in res[0]["answer"]


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "mlkit" in capsys.readouterr().out