# mlkit — ML / Deep Learning / LLM algorithms, from scratch in NumPy

A single CLI (`mlkit`) that implements, **from scratch in pure NumPy**
(no scikit-learn, no PyTorch), the full stack of algorithms from
*traditional machine learning → deep learning → large language models*:

| Layer | What's implemented |
|---|---|
| **Machine learning** | Linear/Ridge/Lasso/ElasticNet, Logistic Regression, SVM (sub-gradient + SMO kernels), k-NN, Naive Bayes, Decision Tree (CART), Random Forest, Gradient Boosting, XGBoost (Newton), K-Means, DBSCAN, Hierarchical (Ward etc.), GMM (EM), PCA, t-SNE, UMAP, LDA |
| **Deep learning** | a reverse-mode **autograd engine**, MLP, CNN, RNN/GRU/LSTM, attention + multi-head attention, Vision Transformer, GAN, VAE, DDPM diffusion, optimizers (SGD/Adam/AdamW/RMSProp), gradient checking |
| **LLMs** | char + **BPE tokenizers**, decoder-only **Transformer** (CLM & MLM pre-training), decoding (greedy / temperature / top-k / top-p / beam search), **KV cache**, **LoRA** fine-tuning, **INT8/INT4 quantization**, **flash attention**, **speculative decoding**, **RAG** (flat / IVF / HNSW vector search), ReAct / Chain-of-Thought agent loops |

Everything is verifiable: `mlkit dl gradcheck` checks the autograd gradients
of every component against central finite differences.

---

## Install / run

```bash
python3 -m venv .venv && source .venv/bin/activate   # PEP 668-safe
pip install -e .            # installs the `mlkit` console script
mlkit --help

# or without installing:
python -m mlkit --help
```

Only dependency: `numpy`.

---

## 1 · Machine learning — `mlkit ml`

```bash
mlkit data list                                   # built-in datasets
mlkit ml logistic-regression --dataset moons
mlkit ml random-forest --n-estimators 100 --depth 8
mlkit ml xgboost --dataset regression --n-estimators 200
mlkit ml svm --kernel rbf --C 2
mlkit ml knn --k 7 --weights distance
mlkit ml kmeans --k 3 --dataset gmm
mlkit ml dbscan --eps 0.8 --min-samples 5
mlkit ml hierarchical --k 3 --linkage ward
mlkit ml gmm --k 3 --covariance full
mlkit ml pca --components 2
mlkit ml tsne --perplexity 30 --iterations 600
mlkit ml umap --neighbors 15 --min-dist 0.1
mlkit ml lda --components 2
mlkit ml ridge --alpha 0.5 --dataset regression
mlkit ml lasso --alpha 0.05 --dataset regression
mlkit ml elasticnet --alpha 0.05 --l1-ratio 0.5
mlkit ml decision-tree --depth 6 --criterion entropy
mlkit ml gradient-boosting --n-estimators 200 --lr 0.05
mlkit ml naive-bayes --variant gaussian
```

Common flags: `--dataset <name>` or `--csv file.csv --target col`,
`--n`, `--test`, `--seed`, `--json` (machine-readable output).

Compare every supervised learner in one table:

```bash
mlkit bench --dataset circles
```

```
algorithm          test accuracy  fit time
─────────────────  ─────────────  ────────
logistic-reg       0.4133         35.1 ms     ← linear model fails on rings
linear-svm         0.4933         57.4 ms
kernel-svm         0.9867         238.6 ms    ← RBF kernel succeeds
knn(k=5)           0.9867         0.0 ms
naive-bayes        0.9533         0.1 ms
decision-tree      0.9733         7.5 ms
random-forest      0.9800         257.8 ms
gradient-boosting  0.9533         701.1 ms
xgboost            0.9733         79.7 ms
```

---

## 2 · Deep learning — `mlkit dl`

The engine lives in `mlkit/dl/tensor.py`: a from-scratch reverse-mode
autograd over NumPy arrays (broadcasting-aware, vectorised backward passes)
with ops for arithmetic, reductions, softmax/log-softmax, fused
cross-entropy, im2col convolution, max/avg pooling, gather (embeddings),
dropout, concatenation… `mlkit/dl/nn.py` builds layers on top
(Linear, LayerNorm, Conv2d, RNN/GRU/LSTM, multi-head attention with causal
masking + KV cache, Transformer blocks), and `mlkit/dl/optim.py` implements
SGD (momentum/Nesterov), Adam, AdamW and RMSProp.

```bash
mlkit dl gradcheck        # verify autograd vs finite differences (10/10 PASS)
mlkit dl mlp --dataset spirals --epochs 150 --lr 0.005 --hidden 64,64
mlkit dl cnn --epochs 15 --channels 8,16          # images: circle/square/triangle/cross
mlkit dl rnn --cell lstm --dataset copy --epochs 30
mlkit dl rnn --cell gru --dataset sine --epochs 40   # sequence regression
mlkit dl vit --depth 2 --patch 4 --epochs 12        # Vision Transformer
mlkit dl gan --epochs 800                          # 2-D generative demo
mlkit dl vae --latent 2 --epochs 400
mlkit dl diffusion --T 100 --schedule cosine --epochs 800
mlkit dl optim --function rosenbrock --steps 400    # optimizer comparison
```

Optimizer comparison (`mlkit dl optim`) prints each optimizer's loss curve
on the Rosenbrock banana function:

```
optimizer  start    final     best      <1e-3 @  loss curve
sgd        24.2000  0.004576  0.004576  -        24.2000 → 0.0046  █▄▂▁▁▁▁...
adam       24.2000  5.938435  5.938435  -        24.2000 → 5.9384  ███▇▇▇▆...
rmsprop    24.2000  3.813737  3.813737  -        24.2000 → 3.8137  █▇▆▅▄▃▂...
```

---

## 3 · Large language models — `mlkit llm`

`mlkit/llm/` contains a GPT-style decoder-only Transformer trained on a
built-in synthetic English corpus with **both pre-training objectives**
(next-token CLM and masked MLM), plus the whole inference/efficiency stack.

```bash
# 1. tokenizers: character-level and byte-pair encoding (trained from scratch)
mlkit llm tokenize --tokenizer bpe --text "the old king found the map" --show-merges
mlkit llm tokenize --tokenizer char --text "hello world"

# 2. pre-train (clm = GPT-style, mlm = BERT-style) and save a checkpoint
mlkit llm train --out ckpt/ --objective clm --train-steps 400 --corpus 8000
mlkit llm train --out ckpt-mlm/ --objective mlm
mlkit llm evaluate --ckpt ckpt/

# 3. generation: greedy / temperature / top-k / top-p / beam search
mlkit llm generate --ckpt ckpt/ --prompt "the " --tokens 60 --temperature 0
mlkit llm generate --ckpt ckpt/ --prompt "the " --tokens 60 --temperature 1.0 --top-k 5
mlkit llm generate --ckpt ckpt/ --prompt "the " --tokens 60 --temperature 0.9 --top-p 0.9
mlkit llm generate --ckpt ckpt/ --prompt "the " --beam 4

# 4. KV cache
mlkit llm bench-cache --ckpt ckpt/ --tokens 40     # speedup + identical outputs
mlkit llm kv --ckpt ckpt/ --steps 8                # cache growth per decode step

# 5. LoRA fine-tuning (freeze base, train low-rank adapters)
mlkit llm lora --ckpt ckpt/ --rank 4 --targets c_attn,c_proj --steps 200 --out adapter.npz
mlkit llm merge --ckpt ckpt/ --adapter adapter.npz  # fold delta_W into weights

# 6. quantization: int8 per-channel / int4 group-wise
mlkit llm quantize --ckpt ckpt/                    # perplexity + memory table

# 7. flash attention (tiled online softmax) vs naive
mlkit llm flash --seq-len 256 --heads 8

# 8. speculative decoding (draft model + target verification)
mlkit llm speculative --ckpt ckpt/ --k 4 --tokens 32

# 9. RAG: chunk → embed → vector index → retrieve → answer
mlkit llm rag --index hnsw --query "how does speculative decoding work?"
mlkit llm rag --index ivf --embedder tfidf --query "what is lora"
mlkit llm rag --generate --query "what is rlhf"    # prompt-stuffed generation

# 10. agent loops (Thought → Action → Observation → Answer)
mlkit llm agent --mode react
mlkit llm agent --mode cot --question "calculate 12 * (8 + 3) - 45 / 5"
```

Representative output:

```
$ mlkit llm train --out ckpt/ --train-steps 400
    model: 104,384 params (2,368 tied)  objective: next-token (CLM)
    step   399  train 0.37  val 0.59  ppl 1.80   █▄▂▁▁

$ mlkit llm quantize --ckpt ckpt/
bits  ppl fp32  ppl quant  bytes fp32  bytes quant  compression  max err
int8  1.561     1.559      402688      104404       3.86x        0.0033
int4  1.561     1.858      402688      56628        7.11x        0.0597

$ mlkit llm bench-cache --ckpt ckpt/
  no_cache_ms 93.7  |  kv_cache_ms 25.8  |  speedup 3.64x  |  identical outputs: True

$ mlkit llm lora --ckpt ckpt/ --rank 4
    trainable 2,048 / 106,432 params (1.92%)  rank=4 alpha=1.0
```

---

## Project layout

```
mlkit/
├── cli.py               # argparse CLI (data / ml / bench / dl / llm)
├── data.py              # synthetic datasets, CSV loading, splits, corpus
├── metrics.py           # accuracy/precision/recall/F1, MSE/R², silhouette…
├── utils.py             # tables, ASCII loss curves, timing
├── ml/                  # classical machine learning
│   ├── linear.py        #   OLS, Ridge (closed form), Lasso/ElasticNet (CD)
│   ├── logistic.py      #   softmax regression, mini-batch GD
│   ├── svm.py           #   Pegasos-style linear SVM + Platt SMO kernels
│   ├── knn.py           #   brute-force k-NN (euclidean/manhattan, weighted)
│   ├── naive_bayes.py   #   Gaussian + Multinomial NB
│   ├── tree.py          #   CART decision trees + Random Forest (bagging)
│   ├── boosting.py      #   Friedman GBM + XGBoost-style Newton boosting
│   ├── cluster.py       #   K-Means(++), DBSCAN, hierarchical (Lance-Williams), GMM-EM
│   └── decomposition.py #   PCA (SVD), t-SNE, UMAP (fuzzy graph + SGD), Fisher LDA
├── dl/                  # deep learning
│   ├── tensor.py        #   autograd engine + ops + gradcheck utilities
│   ├── nn.py            #   layers: Linear/Conv/Pool/RNN/GRU/LSTM/MHA/Blocks…
│   ├── optim.py         #   SGD, Adam, AdamW, RMSProp + grad clipping
│   └── train.py         #   MLP/CNN/Sequence/ViT/GAN/VAE/diffusion trainers
└── llm/                 # large language models
    ├── tokenizer.py     #   CharTokenizer + BPE (train/encode/decode/save)
    ├── model.py         #   TinyGPT: decoder-only Transformer, CLM + MLM
    ├── generate.py      #   greedy, temperature, top-k, top-p, beam, KV cache
    ├── lora.py          #   LoRALinear, freeze/apply/save/load/merge
    ├── quantize.py      #   INT8 per-channel, INT4 group-wise, ppl eval
    ├── flash.py         #   flash attention (online softmax) + benchmark
    ├── speculative.py   #   draft/target verification sampling
    ├── rag.py           #   chunking, TF-IDF/LM embedders, flat/IVF/HNSW, RAG
    ├── agent.py         #   ReAct + CoT loops with calculator/convert/search tools
    └── checkpoint.py    #   save/load model+tokenizer+config
```

---

## Verification

```bash
mlkit dl gradcheck        # 10/10: every backward pass vs finite differences
python -m pytest tests/ -q   # unit + CLI smoke tests
```

Gradient checks cover: elementwise chains, cross-entropy, conv/pooling,
embeddings, concat/stack, softmax/log-softmax, attention (with causal mask),
LayerNorm, full multi-head attention & Transformer blocks, and the VAE
(reparameterisation + KL) — all matching central finite differences within
float32 tolerance.

## References

- [Scikit-learn: Choosing the right estimator](https://scikit-learn.org/stable/tutorial/machine_learning_map/)
- Goodfellow, Bengio, Courville — [Deep Learning](https://www.deeplearningbook.org/)
- Vaswani et al. — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
- [Sebastian Raschka: ML algorithm overviews](https://sebastianraschka.com/blog/)
- Hu et al. — [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Leviathan et al. — [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- Dao et al. — [FlashAttention](https://arxiv.org/abs/2205.14135)
- Johnson, Douze, Jégou — [Billion-scale similarity search with GPUs (FAISS)](https://arxiv.org/abs/1702.08734)
- Malkov, Yashunin — [Efficient and robust approximate nearest neighbor search (HNSW)](https://arxiv.org/abs/1603.09320)
- Yao et al. — [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
