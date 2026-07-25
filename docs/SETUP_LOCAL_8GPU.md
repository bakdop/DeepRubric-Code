# Running the question+rubric pipeline on a standalone 8-GPU box

Written for an 8×A800-80G machine with no slurm. Everything here is what the
upstream README leaves out — the released assets are **not** sufficient to run
`recursive_qa_agent_v4.py` as documented.

## What upstream is missing

| Upstream says | Reality |
| --- | --- |
| `$WIKI_RETRIEVER_ROOT/e5.index/e5_Flat.index` | **Not in the HF dataset.** Only 3 jsonl files ship. You must build the index yourself. |
| `$WIKI_RETRIEVER_ROOT/e5-base-v2/` | **Not in the HF dataset.** Pull `intfloat/e5-base-v2` separately. |
| `wikilinks.json` | The release ships `wikilinks.jsonl`. It is actually a **single-line JSON dict**, so `json.load` works — only the filename differs. |
| — | `recursive_qa_agent_v4.py:611-614` loads the *entire* `wiki_webpages.jsonl` into a dict. At full scale that is 26.6GB of JSON → 100GB+ RAM. |

`tools/prep_wiki_subset.py` and `tools/build_e5_index.py` in this repo fill those gaps.

## Hardware notes for A800

- 8×80GB = 640GB. A bf16 `Qwen3.5-122B-A10B` is ~234GB, so `TP_SIZE=2` or `4` is
  comfortable. A 35B-A3B (~67GB) runs at `TP_SIZE=1`.
- **A800 is Ampere (sm80): bf16 only, no fp8.** Do not pass `--quantization fp8`.
- **`ninja` must be installed.** Qwen3.5's GDN (gated delta net) linear-attention
  kernels JIT-compile at load time and shell out to `ninja`. Without it every TP
  worker dies with `FileNotFoundError: [Errno 2] No such file or directory:
  'ninja'` and the engine never initialises — it looks like a model-support
  problem but is just a missing build tool. `01_setup_env.sh` installs it and
  prints its path. Note the tool is looked up **on PATH**, so installing the
  package is not enough if you invoke python by absolute path without activating
  the venv; `config.sh` prepends `$DR_VENV/bin` to PATH for exactly this reason.
- **Lower `max_num_seqs`.** Qwen3.5's GDN linear attention is Mamba-like, so each
  concurrent decode sequence needs one Mamba cache block. With the weights
  resident, vLLM's default `max_num_seqs=1024` does not fit and engine init dies
  with `max_num_seqs (1024) exceeds available Mamba cache blocks (N) ... CUDA
  graph capture cannot proceed`. `MAX_NUM_SEQS` defaults to 64 here, which is
  ample: tree expansion is a sequential DFS, so concurrency is 1 per tree and at
  most `asyncio.Semaphore(32)` across trees.
- Qwen3.5 is a hybrid linear-attention MoE. `01_setup_env.sh` prints whether your
  vLLM build registers `Qwen3_5MoeForConditionalGeneration` and what compute
  capability each GPU reports — check that line before assuming the model serves.
  If the linear-attention kernels turn out to need sm90, fall back to a plain
  dense/MoE model (e.g. `Qwen3-30B-A3B`) by setting `MODEL_PATH`.

## Where the data comes from, and how to not download 61GB

| Source | HF repo | Size |
| --- | --- | --- |
| Wikipedia | `inclusionAI/ASearcher-Local-Knowledge` (dataset) | 61 GB — 23.5 + 26.6 + 10.8 |
| OpenScholar | `OpenSciLM/OpenScholar-DataStore-V3` (dataset) | 693 GiB |
| e5 encoder | `intfloat/e5-base-v2` (model) | 2.1 GB whole repo, **439 MB** actually needed |

The e5 repo carries onnx/openvino/`pytorch_model.bin` duplicates of the same
weights; `02_download_data.sh` `--include`s only the six files that matter.

**On a bandwidth-capped cluster, copy the derived subset instead of downloading
the raw corpus.** The subset is what the pipeline actually reads:

| What | Size |
| --- | --- |
| `wiki_corpus.jsonl` + `wiki_webpages.jsonl` + `wikilinks.json` | 2.9 GB |
| minimal e5 encoder | 0.44 GB |
| **subtotal — rebuild the index locally** | **≈3.4 GB** |
| `e5.index/e5_Flat.index`, if you would rather skip the GPU minutes | +4.0 GB |

That is ~18× less traffic than pulling 61 GB and then building the subset yourself.

```bash
# on the target box
SRC=user@host:/path/to/wiki-assets bash scripts/local/02b_import_subset.sh
bash scripts/local/03_build_index.sh --index-only     # minutes on 8 GPUs
# or: SRC=... WITH_INDEX=1 bash scripts/local/02b_import_subset.sh  (no GPU work)
```

### Mirrors (preferred over a proxy)

`config.sh` defaults to mirrors, so no proxy is needed:

```bash
export HF_ENDPOINT=https://hf-mirror.com                       # HuggingFace
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple  # PyPI (Tsinghua TUNA)
```

`HF_ENDPOINT` is honoured by `huggingface_hub`, the `hf` CLI, and `hf_transfer`
alike; `01_setup_env.sh` passes `PIP_INDEX_URL` to `uv pip install --index-url`.
If your site runs its own HF mirror, point `HF_ENDPOINT` at that instead. Tsinghua
mirrors PyPI/conda but not HuggingFace, which is why the two settings differ.

`HF_ENDPOINT` is read at **import** time, so export it before starting the
process — changing it mid-run has no effect.

### `httpx.RemoteProtocolError: Server disconnected without sending a response`

Recent `huggingface_hub` uses `httpx` (older versions used `requests`; both read
`HTTP_PROXY`/`HTTPS_PROXY`). This particular traceback ends in `api.repo_info()`,
i.e. the **metadata** call that runs before any file transfer, and the connection
was dropped at the TCP/TLS layer rather than answered with a status code. That
means the host was never reached. In order of likelihood:

1. `HF_ENDPOINT` is unset, so it is talking to `huggingface.co` directly.
   `echo $HF_ENDPOINT` — if empty, export the mirror and retry.
2. A stale proxy variable points somewhere dead:
   `unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy`.
3. The mirror itself is flaky — retry, or switch `HF_ENDPOINT` to a site mirror.

### `httpx.ReadTimeout: The read operation timed out`

Different failure: the host *was* reached, it just did not answer in time.
`huggingface_hub` defaults both timeouts to **10s**, which a cross-border hop to a
mirror routinely exceeds. `config.sh` raises them and turns off the telemetry
request that `hf_hub >= 1.24` issues alongside the real one (it shows up in the
traceback as `_detect_agent.py:_fetch_registry` and is unrelated to your download):

```bash
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
```

`02_download_data.sh` also retries each repo up to `HF_RETRIES` (default 5) times
with a 30s backoff, which costs nothing because `hf download` resumes.

Check reachability directly; both should answer:

```bash
curl -sI https://hf-mirror.com | head -1
curl -s https://hf-mirror.com/api/datasets/inclusionAI/ASearcher-Local-Knowledge | head -c 200
```

`hf_transfer` is **enabled** by default — it parallelises range requests and is a
large speedup on a direct link. It does *not* reliably honour `HTTP(S)_PROXY`, so
if you are forced through a proxy, set `HF_HUB_ENABLE_HF_TRANSFER=0`.

`hf download` resumes from partial files, so an interrupted transfer just needs the
same command again.

**OpenScholar is 693 GiB and is only needed for the `recursive_qa_agent_v44`
branch.** Its `/search` service additionally needs the `embeddings/*.pkl` shards
and a built index — downloading only `passages/` gets you root sampling but no
working retrieval. On a metered link, skip this branch entirely.

## Disk budget

| Item | Size |
| --- | --- |
| Wikipedia raw (`ASearcher-Local-Knowledge`) | 61 GB |
| Wikipedia subset @ `KEEP_ONE_IN=20` + index | ~5 GB |
| Wikipedia **full** corpus + Flat index | ~24 GB + ~80 GB |
| OpenScholar DataStore V3 (optional) | 693 GiB |
| python env (vLLM + CUDA wheels) | ~30 GB |

## Steps

```bash
git clone -b bl4363/pipeline-run https://github.com/bakdop/DeepRubric-Code.git
cd DeepRubric-Code

# Everything is configured through scripts/local/config.sh; override via env.
export DR_DATA=/data/deeprubric            # where corpora live
export DR_VENV=/data/deeprubric-venv
export MODEL_PATH=/path/to/Qwen3.5-122B-A10B
export TP_SIZE=4

bash scripts/local/01_setup_env.sh         # venv + vllm; prints GPU/arch check
bash scripts/local/02_download_data.sh     # 61GB; WITH_OPENSCHOLAR=1 adds 693GiB
bash scripts/local/03_build_index.sh       # subset + 8-GPU index build + merge
bash scripts/local/04_serve.sh             # vLLM + retriever, stays in foreground
# ... in a second shell:
MAX_DEPTH=1 bash scripts/local/05_generate.sh   # fast plumbing check
MAX_DEPTH=3 bash scripts/local/05_generate.sh   # the shipped default
```

### Scale knobs

`KEEP_ONE_IN` controls the subset. Sampling is at **article** level (every passage
of a kept article is kept) so retrieval inside the subset still supports multi-hop
drilling on kept topics — a flat random passage sample would not.

```bash
KEEP_ONE_IN=20 bash scripts/local/03_build_index.sh   # ~1.3M passages, 4GB index (default)
KEEP_ONE_IN=1 MAX_PASSAGES=999999999 \
  bash scripts/local/03_build_index.sh                # full ~26M passages, ~80GB index
```

Full scale also needs ~100GB RAM for the retriever's `PageAccess` dict and for
`recursive_qa_agent_v4.py`'s own pages dict.

## The invariant that will silently ruin your data

`local_retrieval_server.py` does `load_docs(corpus, idxs)` — a FAISS row number is
used **directly as a corpus line number**. The index stores only vectors, never
text. If the index and `wiki_corpus.jsonl` are one line out of sync, retrieval
returns unrelated passages, and nothing raises: you just get quietly garbage
evidence, and the generated rubrics will be grounded in the wrong facts.

Both `build_e5_index.py` and `merge_faiss_shards.py` therefore assert
`corpus lines == index rows` at the end. **Do not skip that check**, and rebuild
the index whenever you regenerate the corpus subset.

The multi-GPU build shards by **contiguous** line ranges, so concatenating shards
in id order reproduces corpus order. A round-robin split would not.

## Encoder contract

`build_e5_index.py` mirrors `Encoder` in the vendored server exactly:
`"passage: "` prefix, mean pooling over the attention mask, L2 normalisation,
`IndexFlatIP` (inner product on normalised vectors == cosine). Queries get
`"query: "` at serve time. Changing any one of these on only one side degrades
retrieval without any error.

## Model roles

The two `--*-model` flags are separate on purpose (upstream defaults in brackets):

| Flag | Used for | Upstream default |
| --- | --- | --- |
| `--extract-model` | root summary, per-node evidence extraction, child-query proposal — runs at every node | `Qwen/Qwen3.5-35B-A3B-Instruct` (local) |
| `--qa-model` | branch gate, and the single `BASE_QA_TREE_PROMPT` call that produces the question + rubrics | `deepseek/deepseek-chat` (API) |

`OpenAIAPIClient` reads a single `OPENAI_BASE_URL`, so upstream's config implies a
multi-model gateway. `05_generate.sh` points both roles at one local server; to
reproduce upstream, put a LiteLLM-style proxy in front and set the two model names
to whatever that proxy routes.

## Expected cost per sample

Defaults are `max_depth=3`, `root_children=6`, `mid_children=4`, `max_children=3`,
`max_nodes=50`. Expansion is a **sequential DFS** — no concurrency inside one tree,
only across trees (`asyncio.Semaphore(32)`). Per sample that is roughly:

- 1 retrieval per node (topk=5), ~50 nodes
- 2 LLM calls per node (extract + propose), plus 1 gate call per non-leaf
- 2 root seeding calls + 1 final synthesis call
- ≈ 110-150 LLM calls total

Because the node budget is a single global counter checked during DFS, the tree
comes out **lopsided**: the first depth-1 subtrees expand fully and later siblings
degenerate into leaves once `node_count >= max_nodes`.
