# Minimal CUDA `llama-server` build

Target: Linux, CUDA 13.1 at `/usr/local/cuda-13.1`, GPUs of compute capability 75 and 86,
`ccache` installed. Build directory `build-reasoning-temp`, shared libs, server binary only.

The ready-to-run script is `scripts/build-server-cuda.sh`.

## 1. Flag mapping from an upstream llama.cpp command line

The root `CMakeLists.txt:26` sets `CMAKE_WARN_UNUSED_CLI YES`, so any `-D` the project does
not read is reported at the end of configure. Nothing is silently swallowed.

| Upstream flag | BeeLlama equivalent | Note |
|---|---|---|
| `CMAKE_BUILD_TYPE=Release` | same | `CMakeLists.txt:36`. Already the non-MSVC default. |
| `BUILD_SHARED_LIBS=ON` | same | `CMakeLists.txt:91`. Already the Linux default. |
| `GGML_CUDA=ON` | same | `ggml/CMakeLists.txt:199`. |
| `GGML_CUDA_FA=ON` | same | `ggml/CMakeLists.txt:205`, default ON. OFF adds `GGML_CUDA_NO_FA`. Required for KVarN attention. |
| `GGML_CUDA_FA_ALL_QUANTS=OFF` | same | `ggml/CMakeLists.txt:206`, default OFF. In this fork it drives **both** the standard vec matrix and the KVarN fast-decode matrix. See section 2. |
| `GGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16"` | **drop** | Does not exist anywhere in the tree. The pair list is not user-selectable. All four requested pairs are already in the default 50. |
| `GGML_CUDA_FORCE_CUBLAS=OFF` | same | `ggml/CMakeLists.txt:202`, applied at `ggml/src/ggml-cuda/CMakeLists.txt:158`. |
| `GGML_CUDA_FORCE_MMQ=OFF` | same | `ggml/CMakeLists.txt:201`, applied at `ggml/src/ggml-cuda/CMakeLists.txt:154`. |
| `GGML_CUDA_GRAPHS=ON` | same | `ggml/CMakeLists.txt:210`. Default ON from the llama root (`CMakeLists.txt:185`). Defines `GGML_CUDA_USE_GRAPHS`. |
| `GGML_CUDA_NCCL=ON` | same | `ggml/CMakeLists.txt:211`, default ON. If NCCL is absent you get `message(STATUS "Warning: NCCL not found…")` at `ggml/src/ggml-cuda/CMakeLists.txt:210` and configure continues. |
| `GGML_CUDA_NO_PEER_COPY=OFF` | same | `ggml/CMakeLists.txt:203`. |
| `GGML_CUDA_NO_VMM=OFF` | same | `ggml/CMakeLists.txt:204`. OFF keeps the `CUDA::cuda_driver` link (`ggml/src/ggml-cuda/CMakeLists.txt:198`). |
| `GGML_CUDA_COMPRESSION_MODE=size` | same | `ggml/CMakeLists.txt:212`, cache STRING over `none;speed;balance;size`, default already `size`. With CUDA >= 12.8 it becomes `-compress-mode=size` (`ggml/src/ggml-cuda/CMakeLists.txt:229`). |
| `GGML_NATIVE=ON` | same | `ggml/CMakeLists.txt:123`. Does not fight the explicit arch list: the `native` CUDA-arch autodetect at `ggml/src/ggml-cuda/CMakeLists.txt:27` only runs when `CMAKE_CUDA_ARCHITECTURES` is undefined. |
| `GGML_CPU=ON` | same | `ggml/CMakeLists.txt:189`, default ON. |
| `GGML_LLAMAFILE=ON` | same | `ggml/CMakeLists.txt:197`. Default ON from the llama root. |
| `GGML_CCACHE=ON` | same | `ggml/CMakeLists.txt:125`, default ON. Finds `ccache` or `sccache`; no-op if neither exists. |
| `LLAMA_BUILD_SERVER=ON` | same | `CMakeLists.txt:148`, consumed at `tools/CMakeLists.txt:24`. Gates `tools/ui`, `tools/cli`, `tools/server` only. |
| `LLAMA_BUILD_TOOLS=ON` | same | `CMakeLists.txt:146`, used at `CMakeLists.txt:259`. Required, since the server lives under `tools/`. |
| `LLAMA_BUILD_APP=ON` | **set OFF** | `CMakeLists.txt:149`. `app/llama-app` links eight tool `*-impl` libraries (`app/CMakeLists.txt:6-15`). Off is what makes a server-only build possible. |
| `LLAMA_USE_PREBUILT_UI=ON` | same | `CMakeLists.txt:151`, consumed in `tools/ui/CMakeLists.txt:53` -> `scripts/ui-assets.cmake`. See section 4. |
| `LLAMA_UI_GZIP=ON` | same | `tools/ui/CMakeLists.txt:4`, cache BOOL, default ON. Only defined once `tools/ui` is added, i.e. needs tools + server on. |
| `LLAMA_BUILD_TESTS=OFF` | same | `CMakeLists.txt:145`. |
| `LLAMA_BUILD_EXAMPLES=OFF` | same | `CMakeLists.txt:147`. Also skips `pocs/`. |
| `LLAMA_BUILD_MTMD=OFF` | **drop** | `CMakeLists.txt:272`. Not a "disable multimodal" switch; it is a standalone libmtmd packaging hook that only fires when `NOT (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS)`. Default is already OFF. Harmless no-op, saves nothing. |
| `LLAMA_OPENSSL=ON` | same | `CMakeLists.txt:157`, default ON, consumed at `vendor/cpp-httplib/CMakeLists.txt:128`. Missing OpenSSL is a warning, not an error. |
| `CMAKE_CUDA_ARCHITECTURES="75;86"` | same | Honored verbatim; the `12X -> 12Xa` rewrite at `ggml/src/ggml-cuda/CMakeLists.txt:80-92` leaves 75/86 alone. Note `GGML_CUDA_ARCH` is rejected with FATAL_ERROR at `CMakeLists.txt:30`, so this is the right variable. |
| `CUDAToolkit_ROOT=/usr/local/cuda-13.1` | same | Read by `find_package(CUDAToolkit)` at `ggml/src/ggml-cuda/CMakeLists.txt:3`. |
| `CMAKE_CUDA_COMPILER=.../nvcc` | same | Honored by `enable_language(CUDA)` at `ggml/src/ggml-cuda/CMakeLists.txt:59`. |
| `CUDA_cudart_LIBRARY`, `CUDA_cublas_LIBRARY`, `CUDA_cublasLt_LIBRARY` | **drop** | Legacy `FindCUDA` variables, never read. The tree links the modern imported targets `CUDA::cudart`, `CUDA::cublas`, `CUDA::cuda_driver` at `ggml/src/ggml-cuda/CMakeLists.txt:174-202`. `CUDA::cublasLt` only appears in the `GGML_STATIC` branch (line 183). |
| `CMAKE_INSTALL_RPATH="...;$ORIGIN"` | **fix quoting** | Works, but inside double quotes bash expands `$ORIGIN` to the empty string, leaving `"/usr/local/cuda-13.1/lib64;"`. Use single quotes. The project itself sets no RPATH anywhere, so this is entirely yours. |
| `CMAKE_BUILD_WITH_INSTALL_RPATH=ON` | same | With `BUILD_SHARED_LIBS=ON` this makes build-tree binaries use the install RPATH, so `build-reasoning-temp/bin/llama-server` only finds its sibling `libggml*.so` if `$ORIGIN` survived the quoting above. |

Two additions worth making explicitly:

| Added flag | Why |
|---|---|
| `GGML_CUDA_KVARN=ON` | `ggml/CMakeLists.txt:207`, default ON. Master switch for `kvarn.cu`, `kvarn-wht.cu` and every `fattn-mma-kvarn*` instance (`ggml/src/ggml-cuda/CMakeLists.txt:106-110` and the filter in `ggml_cuda_select_kvarn_fast_decode_sources`). Passing it explicitly stops a stale cache from disabling KVarN silently. |
| `LLAMA_BUILD_UI=OFF` | `CMakeLists.txt:150`, already the default. Explicit so the npm build is never attempted. |

Removed options that a stale cache might still carry: `GGML_CUDA_KVARN_FA` and
`GGML_CUDA_KVARN_FAST_DECODE_ALL_PAIRS` are actively `unset(... CACHE)`d at
`ggml/CMakeLists.txt:208-209`; `GGML_CUDA_FA_HALF_QUANTS` no longer exists.

## 2. How BeeLlama picks the compiled FlashAttention pairs

There is no per-pair selection. One flag, `GGML_CUDA_FA_ALL_QUANTS`, drives two
independent matrices.

### 2.1 The 15 default KVarN bit pairs

`GGML_CUDA_KVARN_DEFAULT_PAIRS` is a plain `set()` at `ggml/CMakeLists.txt:216-232`
(not a cache entry, so `-D` cannot override it), guarded by an `EQUAL 15` FATAL_ERROR at
`ggml/CMakeLists.txt:235`:

```
k8-v8  k8-v6  k8-v5
k6-v6  k6-v5  k6-v4
k5-v5  k5-v4  k5-v3
k4-v4  k4-v3  k4-v2
k3-v3  k3-v2
k2-v2
```

The rule: K bits >= V bits, at most two steps apart in the ordered widths
{2, 3, 4, 5, 6, 8}. `ggml_cuda_select_kvarn_fast_decode_sources()`
(`ggml/CMakeLists.txt:240`) turns each into
`template-instances/fattn-mma-kvarn-decode-instance-kN-vM.cu`, and FATAL_ERRORs if the
selected count does not match. All 36 ordered pairs exist on disk; CMake picks 15 or 36.

The runtime mirror of exactly this rule is
`ggml_cuda_fattn_kvarn_fast_decode_pair_enabled()` at
`ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu:479-493`, plus the
`GGML_CUDA_FATTN_KVARN_FAST_DECODE_DISPATCH_K` macro at line 497.

`GGML_CUDA_FA_ALL_QUANTS=ON` adds the remaining 21:

```
V wider than K (15):
  k2-v3 k2-v4 k2-v5 k2-v6 k2-v8
  k3-v4 k3-v5 k3-v6 k3-v8
  k4-v5 k4-v6 k4-v8
  k5-v6 k5-v8
  k6-v8
K more than two steps wider than V (6):
  k5-v2  k6-v3  k6-v2  k8-v4  k8-v3  k8-v2
```

### 2.2 The 50 default standard vec pairs

`ggml_cuda_get_fattn_vec_default_pairs()` (`ggml/CMakeLists.txt:277-320`) derives the
standard quant matrix from the *same* 15 bit pairs, mapping bit widths onto concrete
cache types:

```
8 -> q8_0
6 -> q6_1, q6_0
5 -> q5_1, q5_0
4 -> q4_1, q4_0
3 -> q3_1, q3_0
2 -> q2_1, q2_0s      (q2_0s = GGML_TYPE_Q2_0S, Bee's cache-facing q2_0)
```

For each bit pair it takes the full K-types x V-types cross product, skipping same-bit
combinations where the K variant index is greater than the V variant index
(`ggml/CMakeLists.txt:306`) - so within one width only `_1:_1`, `_1:_0` and `_0:_0`
survive, and `_0:_1` is dropped. That gives 48 quantized pairs; the two homogeneous
float tails `f16:f16` and `bf16:bf16` are seeded at `ggml/CMakeLists.txt:286` for a total
of exactly 50, asserted at `ggml/CMakeLists.txt:315`. Each pair is then added by name via
`ggml_add_fattn_vec_pair()` (`ggml/src/ggml-cuda/CMakeLists.txt:121-143`), which
FATAL_ERRORs on a missing instance file.

The four pairs the upstream command asked for are all in the default set:
`q4_0:q4_0` (from `k4-v4`), `q8_0:q8_0` (from `k8-v8`), `f16:f16` and `bf16:bf16`
(the float tails). The instance files
`template-instances/fattn-vec-instance-{q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16}.cu` exist.

`GGML_CUDA_FA_ALL_QUANTS=ON` instead globs all 169 `fattn-vec-instance-*.cu` and adds
`add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)`
(`ggml/src/ggml-cuda/CMakeLists.txt:131-134`). The runtime mirror is
`ggml_cuda_fattn_pair_compiled()` / `ggml_cuda_fattn_default_quant_pair()` at
`ggml/src/ggml-cuda/fattn.cu:449-489`, plus the checked-in
`ggml/src/ggml-cuda/fattn-vec-dispatch.cuh`. An uncompiled pair is reported as such and
routed to the non-vec FA path; it is never silently wrong.

Narrowing the list to four pairs is not expressible as a flag. It would require editing
`ggml/CMakeLists.txt`, relaxing two FATAL_ERROR guards, regenerating
`fattn-vec-dispatch.cuh` via `scripts/gen-fattn-vec-dispatch.py`, and editing the runtime
predicate in `fattn.cu`. The practical choice is binary: default 50 + 15, or 169 + 36.

### 2.3 kvarn4 specifically

KVarN pairs are addressed by bit width only, so what matters is `k4` / `v4`.

Fast-decode (`decode-split` / `decode-vector`) out of the box, no extra flag:

- `K=kvarn4 V=kvarn4` (`k4-v4`)
- `K=kvarn4 V=kvarn3` (`k4-v3`)
- `K=kvarn4 V=kvarn2` (`k4-v2`)
- `K=kvarn5 V=kvarn4` (`k5-v4`)
- `K=kvarn6 V=kvarn4` (`k6-v4`)

Falls back to the descriptor-native generic MMA kernel unless
`GGML_CUDA_FA_ALL_QUANTS=ON`:

- `k4-v5`, `k4-v6`, `k4-v8`, `k8-v4`, `k3-v4`, `k2-v4`

The fallback is still correct and still consumes KVarN records directly. What it gives up
is the specialized decode kernel: occupancy-based 64/128-token geometry, division-free
single-stream scan, packed fragment loads, K/V reuse across query tiles, and
query-count-independent split partitioning. It is a single-token decode throughput
regression only. Prefill takes `prompt-generic-mma` either way, so prefill is unaffected.

The standard vec matrix (50 vs 169) is not consumed by the KVarN path at all - it is for
`--cache-type-k q4_0` style caches. `GGML_CUDA_FA_ALL_QUANTS=OFF` therefore costs nothing
for `kvarn4 / kvarn4`.

## 3. Server-only build

`LLAMA_BUILD_TOOLS=ON` is mandatory (the server is `tools/server`), and
`tools/CMakeLists.txt:17-44` adds unconditionally: `batched-bench`, `gguf-split`,
`imatrix`, `llama-bench`, `completion`, `perplexity`, `quantize`, `tokenize`, `tts`,
`mtmd`, `fit-params`, `results`, plus `cvector-generator` and `export-lora` when
`NOT GGML_BACKEND_DL AND GGML_CPU`, plus `rpc` under `GGML_RPC` and `tuning` under
`GGML_METAL`. Only `ui`, `cli` and `server` sit behind `LLAMA_BUILD_SERVER`. **No flag
combination builds only the server.**

So do both:

1. `-DLLAMA_BUILD_APP=OFF` keeps `app/llama-app` and its eight `*-impl` dependencies out
   of the configure entirely (`CMakeLists.txt:263`, `app/CMakeLists.txt:6-15`).
2. `cmake --build ... --target llama-server` compiles only that target's dependency
   closure: `ggml-base`, `ggml-cpu`, `ggml-cuda`, `ggml`, `llama`, `llama-common`,
   `mtmd`, `cpp-httplib`, `llama-ui` (+ `llama-ui-assets`), `server-context`,
   `llama-server-impl`, `llama-server`. Everything else is configured but never built.
   `add_subdirectory()` still runs for the skipped tools, costing a few seconds of
   configure time.

### mtmd cannot be dropped

`tools/server/CMakeLists.txt:36`:

```cmake
target_link_libraries(server-context PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})
```

with `target_include_directories(server-context PRIVATE ../mtmd)` at line 34 and the same
on `llama-server-impl` at line 52. `mtmd` is a hard link dependency of the server in this
tree, and `LLAMA_BUILD_MTMD=OFF` does not change that (see section 1). The only real trim
available is `-DMTMD_VIDEO=OFF` (`tools/mtmd/CMakeLists.txt:5`), which drops the
ffmpeg-subprocess video path.

## 4. Web UI provisioning

`tools/ui/CMakeLists.txt:44` declares `llama-ui-assets` as `add_custom_target(... ALL)`
running `scripts/ui-assets.cmake` at **build** time, so a network-less configure always
succeeds. Provisioning priority (`scripts/ui-assets.cmake:472-534`):

1. Pre-built assets at `tools/ui/dist/index.html`. This directory does not exist in the
   repository, so by default this path does not hit.
2. `BUILD_UI=ON` (from `LLAMA_BUILD_UI`) -> npm build. Off here.
3. `HF_ENABLED=ON` (from `LLAMA_USE_PREBUILT_UI`) -> `file(DOWNLOAD)` of
   `https://huggingface.co/buckets/ggml-org/llama-ui/resolve/<version>/dist.tar.gz` with
   SHA256 verification (`scripts/ui-assets.cmake:422-440`). **This is the path that runs,
   and it needs network at build time.** The candidate list is the resolved build number
   followed by the literal `latest` (`scripts/ui-assets.cmake:414-420`), so a build number
   with no published bucket still resolves; a matching stamp file skips the fetch
   entirely. `HF_TOKEN` helps with rate limits.
4. On failure: `message(WARNING)` and the server is built without an embedded UI. Not
   fatal.

Offline: extract a matching `dist.tar.gz` into `tools/ui/dist/` so priority 1 wins, then
optionally set `-DLLAMA_USE_PREBUILT_UI=OFF`. `ui_validate_assets()`
(`scripts/ui-assets.cmake:65`) FATAL_ERRORs on a truncated tree, so extract the whole
tarball.

`LLAMA_UI_GZIP=ON` gzips each asset into a parallel `_gzip/` tree in the build directory
and embeds those, served with `Content-Encoding: gzip`. Build-time only, no network.

`llama-server-impl` links `llama-ui` and carries `add_dependencies(llama-server-impl
llama-ui-assets)` (`tools/server/CMakeLists.txt:53,55`), so `--target llama-server`
provisions the UI correctly even though `llama-ui-assets` is an `ALL` target.

## 5. Runtime verification

Confirm the cache type was accepted (`src/llama-context.cpp:4664`):

```
llama_context: enabling structured KVarN cache type kvarn4 for target layers [0, 36)
```

The alternative is `cannot enable kvarn4: <reason>; falling back to the normal KV cache`
from `src/llama-context.cpp:4652`.

Then trace the attention route (`ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu:1082-1099`, called
from the route decisions at `:1119-1219`):

```bash
GGML_KVARN_DEBUG_ROUTES=1 build-reasoning-temp/bin/llama-server \
  -m model.gguf --flash-attn on \
  --cache-type-k kvarn4 --cache-type-v kvarn4 -c 4096 --port 8080
```

Each attention op prints one line:

```
kvarn-route backend=CUDA cc=86 wave=32 D=128 k=4 v=4 gqa=8 nq=1 nkv=512 \
  domain=target tail_type=f16 tail_history=0 tail_current=0 \
  route=decode-split entry=direct fallback=none wide_mma=0
```

Route strings, as printed:

| `route=` | Meaning |
|---|---|
| `decode-split`, `decode-vector` | Fast decode path. This is what you want during generation, with `fallback=none`. |
| `generic-mma` | Descriptor-native tiled MMA fallback: correct, slower decode. Means the bit pair is not a compiled fast-decode instance. |
| `prompt-generic-mma` | Prefill. Normal and expected for every pair. |
| `portable-native` | Scalar direct-record path; device lacks the specialized-route capability, or `GGML_KVARN_TEST_FORCE_PORTABLE_FATTN` is set. |
| `materialize-fallback` | KVarN records materialized before attention. The explicit fallback, not the normal route on CUDA. |

The corresponding enum, in descending preference, is in
`ggml/src/ggml-cuda/fattn-kvarn-route-policy.h:21-26`.

## 6. Compile time

`ggml/src/ggml-cuda/template-instances/` holds 301 checked-in, pre-generated `.cu` files.
Neither `template-instances/generate_cu_files.py` nor `scripts/gen-fattn-vec-dispatch.py`
runs at configure or build time; CMake only selects among existing files.

| Family | On disk | Default | `FA_ALL_QUANTS=ON` |
|---|---|---|---|
| `fattn-vec-instance-*` | 169 | 50 | 169 |
| `fattn-mma-kvarn-decode-instance-kN-vM` | 36 | 15 | 36 |
| `fattn-mma-kvarn-instance-ncols1_*-ncols2_*` | 17 | 17 | 17 |
| `fattn-mma-kvarn-decode-combine-instance` | 1 | 1 | 1 |
| `fattn-mma-kvarn-window-common-instance` | 1 | 1 | 1 |
| `fattn-mma-f16-instance-*` | 21 | 21 | 21 |
| `fattn-tile-instance-*` | 12 | 12 | 12 |
| `mmq-instance-*` | 28 | 28 | 28 |
| `mmf-instance-*` | 16 | 16 | 16 |
| **template TUs** | **301** | **161** | **301** |

Plus 71 non-template `.cu` files in `ggml/src/ggml-cuda/` (two of which, `kvarn.cu` and
`kvarn-wht.cu`, are filtered out when `GGML_CUDA_KVARN=OFF`). Total CUDA TUs: 232 default,
372 with `FA_ALL_QUANTS=ON`.

The added files are the expensive ones. A `fattn-vec-instance-X-Y.cu` is four
`DECL_FATTN_VEC_CASE` instantiations (head dims 64/128/256/512), each expanding to several
ncols variants; a `fattn-mma-kvarn-decode-instance-kN-vM.cu` is three
`DECL_FATTN_KVARN_DECODE_CASE` plus one `DECL_FATTN_KVARN_VEC_CASE`. Everything is
compiled once per entry in `CMAKE_CUDA_ARCHITECTURES`, so `75;86` doubles the device-code
work versus a single architecture.

Order-of-magnitude expectation, not a measurement: `FA_ALL_QUANTS=ON` adds 119 vec and 21
KVarN decode instances, roughly 1.9x the translation units and roughly 2 to 2.5x the CUDA
wall clock, plus a substantially larger `libggml-cuda.so`. nvcc peaks at several GB per
job, so on a memory-constrained machine set `JOBS` below the core count. ccache helps on
rebuilds, not on the first build. Per `AGENTS.md`, any concrete timing claim requires an
actual run on the target hardware with the commit recorded.
