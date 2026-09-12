#!/usr/bin/env bash
#
# BeeLlama.cpp - minimal CUDA llama-server build (Linux, CUDA 13.1, sm_75 + sm_86).
#
# Usage:
#   chmod +x build-server-cuda.sh
#   ./build-server-cuda.sh                 # default FA matrix (50 vec pairs + 15 KVarN pairs)
#   FA_ALL_QUANTS=1 ./build-server-cuda.sh # full matrix (169 vec pairs + 36 KVarN pairs)
#   JOBS=8 ./build-server-cuda.sh          # override parallelism
#
# Run from the repository root.
#
# ---------------------------------------------------------------------------
# Flags dropped from the upstream llama.cpp command line, and why
# ---------------------------------------------------------------------------
#
#   -DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16"
#       Does not exist in beellama.cpp (no reference anywhere in the tree).
#       The compiled FlashAttention vec pair list is not user-selectable; it is
#       computed by ggml_cuda_get_fattn_vec_default_pairs() in
#       ggml/CMakeLists.txt:277 and is fixed at 50 pairs (FATAL_ERROR guard at
#       ggml/CMakeLists.txt:315). All four requested pairs - q4_0/q4_0,
#       q8_0/q8_0, f16/f16, bf16/bf16 - are already in that default 50, so
#       nothing is lost by dropping the flag. Because the root CMakeLists sets
#       CMAKE_WARN_UNUSED_CLI YES (CMakeLists.txt:26), passing it would produce
#       a loud "Manually-specified variables were not used" warning.
#
#   -DCUDA_cudart_LIBRARY=... -DCUDA_cublas_LIBRARY=... -DCUDA_cublasLt_LIBRARY=...
#       Legacy FindCUDA variables. The tree uses modern FindCUDAToolkit imported
#       targets (CUDA::cudart, CUDA::cublas, CUDA::cuda_driver) at
#       ggml/src/ggml-cuda/CMakeLists.txt:174-202; those -D names are never read.
#       CUDAToolkit_ROOT plus CMAKE_CUDA_COMPILER already pin the toolkit.
#       (CUDA::cublasLt is only linked in the GGML_STATIC branch, line 183.)
#
#   -DLLAMA_BUILD_MTMD=OFF
#       Not a "disable multimodal" switch. The option is declared at
#       CMakeLists.txt:272 and its add_subdirectory only fires when
#       NOT (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS) - i.e. it is a standalone
#       libmtmd packaging hook. With tools on, tools/CMakeLists.txt:31 adds
#       tools/mtmd unconditionally, and the server hard-links it
#       (tools/server/CMakeLists.txt:36). Passing OFF is a no-op plus a warning.
#
# Changed rather than dropped:
#
#   -DLLAMA_BUILD_APP=ON  ->  OFF
#       app/llama-app links eight tool *-impl libraries (llama-server-impl,
#       llama-cli-impl, llama-completion-impl, llama-bench-impl,
#       llama-batched-bench-impl, llama-fit-params-impl, llama-quantize-impl,
#       llama-perplexity-impl; app/CMakeLists.txt:6-15). Leaving it ON drags all
#       of them into the configure and into "all". Off + --target llama-server
#       is the server-only build.
#
#   -DCMAKE_INSTALL_RPATH="/usr/local/cuda-13.1/lib64;$ORIGIN"
#       Inside double quotes bash expands $ORIGIN to the empty string, so the
#       original line actually set "/usr/local/cuda-13.1/lib64;". Combined with
#       CMAKE_BUILD_WITH_INSTALL_RPATH=ON and BUILD_SHARED_LIBS=ON that leaves
#       build-reasoning-temp/bin/llama-server unable to find its own
#       libggml*.so next to it. Single-quoted below so $ORIGIN reaches CMake.
#
# Added:
#
#   -DGGML_CUDA_KVARN=ON
#       Master switch for the CUDA KVarN kernels and templates
#       (ggml/CMakeLists.txt:207). Default is ON, but passing it explicitly means
#       a stale cache entry cannot silently strip kvarn.cu / kvarn-wht.cu and
#       every fattn-mma-kvarn* instance. Required for kvarn4 to have any
#       dedicated CUDA path at all.
#
#   -DLLAMA_BUILD_UI=OFF
#       Already the default (CMakeLists.txt:150). Explicit so the npm build is
#       never attempted; the prebuilt-UI path is used instead.
#
# ---------------------------------------------------------------------------
# Web UI needs network at BUILD time
# ---------------------------------------------------------------------------
#
#   tools/ui/CMakeLists.txt:44 registers llama-ui-assets as an
#   add_custom_target(... ALL) that runs scripts/ui-assets.cmake at build time,
#   not configure time. With LLAMA_USE_PREBUILT_UI=ON and no tools/ui/dist
#   present (there is none in this tree), it does a file(DOWNLOAD) of
#   https://huggingface.co/buckets/ggml-org/llama-ui/resolve/<version>/dist.tar.gz
#   with SHA256 verification (scripts/ui-assets.cmake:422-440). It tries the
#   resolved build number first and then the literal "latest"
#   (scripts/ui-assets.cmake:414-420), so a build number with no published
#   bucket still gets a UI. A failure of both is a message(WARNING) and the
#   server is built WITHOUT an embedded UI, not a build error. Set HF_TOKEN in
#   the environment if you hit rate limits.
#
#   Offline workaround: extract a matching dist.tar.gz into tools/ui/dist/ so
#   tools/ui/dist/index.html exists. That is provisioning priority 1
#   (scripts/ui-assets.cmake:472) and wins over the download; you can then also
#   pass -DLLAMA_USE_PREBUILT_UI=OFF. The tree validates the dist contents and
#   FATAL_ERRORs on a truncated asset set, so extract the whole tarball.
#
# ---------------------------------------------------------------------------
# Verifying that kvarn4 takes the fast decode path at runtime
# ---------------------------------------------------------------------------
#
#   1. Startup log line from src/llama-context.cpp:4664 confirms the cache type
#      was actually accepted (not silently downgraded):
#
#        llama_context: enabling structured KVarN cache type kvarn4 for target layers [0, 36)
#
#      If instead you see "cannot enable kvarn4: <reason>; falling back to the
#      normal KV cache", the KVarN cache is not in play at all.
#
#   2. Per-op route trace. Set GGML_KVARN_DEBUG_ROUTES=1 and run a short
#      generation; the dispatcher prints one line per attention op from
#      ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu:1082:
#
#        GGML_KVARN_DEBUG_ROUTES=1 build-reasoning-temp/bin/llama-server \
#          -m model.gguf --flash-attn on \
#          --cache-type-k kvarn4 --cache-type-v kvarn4 -c 4096 --port 8080
#
#      Expected during single-token decode:
#
#        kvarn-route backend=CUDA cc=86 ... k=4 v=4 ... route=decode-split entry=direct fallback=none wide_mma=0
#
#      route=decode-split or route=decode-vector with fallback=none is the fast
#      path. route=generic-mma means the pair was not compiled as a fast-decode
#      instance and fell back to the descriptor-native tiled MMA kernel: still
#      correct and still record-native, but it gives up the occupancy-tuned
#      decode geometry. route=prompt-generic-mma during prefill is normal and
#      expected regardless of the pair. route=portable-native or
#      materialize-fallback means something more basic is wrong (device
#      capability or shape).
#
set -euo pipefail

BUILD_DIR="build-reasoning-temp"
CUDA_ROOT="/usr/local/cuda-13.1"
JOBS="${JOBS:-$(nproc)}"

# FA_ALL_QUANTS=1 switches GGML_CUDA_FA_ALL_QUANTS to ON. That takes the
# compiled standard FA vec matrix from 50 to all 169 pairs and the KVarN
# fast-decode matrix from the balanced 15 to all 36 ordered bit pairs
# (+21 pairs, including k4-v5/k4-v6/k4-v8/k8-v4/k3-v4/k2-v4). Cost: +119 vec
# instances and +21 KVarN decode instances, i.e. 161 -> 301 template TUs, each
# compiled for both sm_75 and sm_86. Expect roughly 2 to 2.5x the CUDA
# wall-clock and a substantially larger libggml-cuda.so. Only worth it if you
# actually run an unbalanced K/V pair. k4-v4, k4-v3, k4-v2, k5-v4 and k6-v4 are
# already fast-path in the default build.
if [[ "${FA_ALL_QUANTS:-0}" == "1" ]]; then
    FA_ALL_QUANTS_VALUE=ON
else
    FA_ALL_QUANTS_VALUE=OFF
fi

echo "==> configuring ${BUILD_DIR} (GGML_CUDA_FA_ALL_QUANTS=${FA_ALL_QUANTS_VALUE}, jobs=${JOBS})"

cmake -B "${BUILD_DIR}" -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_KVARN=ON \
    -DGGML_CUDA_FA_ALL_QUANTS="${FA_ALL_QUANTS_VALUE}" \
    -DGGML_CUDA_FORCE_CUBLAS=OFF \
    -DGGML_CUDA_FORCE_MMQ=OFF \
    -DGGML_CUDA_GRAPHS=ON \
    -DGGML_CUDA_NCCL=ON \
    -DGGML_CUDA_NO_PEER_COPY=OFF \
    -DGGML_CUDA_NO_VMM=OFF \
    -DGGML_CUDA_COMPRESSION_MODE=size \
    -DGGML_NATIVE=ON \
    -DGGML_CPU=ON \
    -DGGML_LLAMAFILE=ON \
    -DGGML_CCACHE=ON \
    -DLLAMA_BUILD_COMMON=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=ON \
    -DLLAMA_UI_GZIP=ON \
    -DLLAMA_OPENSSL=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75;86" \
    -DCUDAToolkit_ROOT="${CUDA_ROOT}" \
    -DCMAKE_CUDA_COMPILER="${CUDA_ROOT}/bin/nvcc" \
    -DCMAKE_INSTALL_RPATH='/usr/local/cuda-13.1/lib64;$ORIGIN' \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON

# --target llama-server is what actually makes this a server-only build.
# No flag combination removes the other tools: tools/CMakeLists.txt:17-44 adds
# batched-bench, gguf-split, imatrix, llama-bench, completion, perplexity,
# quantize, tokenize, tts, mtmd, fit-params, results (plus cvector-generator and
# export-lora) unconditionally. Only ui, cli and server sit behind
# LLAMA_BUILD_SERVER. The add_subdirectory() calls still run at configure time,
# but with an explicit target none of those binaries are compiled.
echo "==> building llama-server"
cmake --build "${BUILD_DIR}" --target llama-server -j "${JOBS}"

echo "==> sanity check"
SERVER_BIN="${BUILD_DIR}/bin/llama-server"
if [[ ! -x "${SERVER_BIN}" ]]; then
    echo "ERROR: ${SERVER_BIN} not found or not executable" >&2
    exit 1
fi
"${SERVER_BIN}" --version
echo "==> OK: ${SERVER_BIN}"
