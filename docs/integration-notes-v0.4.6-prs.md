# Integration notes: upstream PRs #25709, #27210 and #24004 on v0.4.6

This is the engineer-facing record of one integration branch. It states what was taken from
upstream, what was changed to fit BeeLlama, what was deliberately left out, how each piece
interacts with KVarN and with owned draft caches, what was actually verified in the integration
environment, and what still needs a GPU and a model. Read it before rebasing any of the three
changes onto a newer base.

## 1. Base and lineage

The branch starts at `78af832652`, the v0.4.6 merge of Anbeeld's BeeLlama.cpp. That release
tracks upstream llama.cpp commit `465e49b9c` (build `b10830`) with ggml 0.23.0. Everything below
is a delta on top of that commit, so any upstream hunk described here was reconciled against the
v0.4.6 base rather than against current upstream master.

The Starwaves1 fork was compared against Anbeeld main and is identical at this point, so there is
no third-party delta to reconcile and nothing fork-specific to preserve from it.

Anbeeld carries an unmerged v0.4.7 branch that adds D64 KVarN support and a set of tail-write
fixes. None of that is included here. The integration was done against v0.4.6 deliberately, so
when v0.4.7 lands the three changes below will need a rebase, and the KVarN-facing parts in
particular (the fingerprint in #24004 and the owned-draft registration in #27210) should be
re-read against the new tail-write behaviour rather than assumed to carry over.

Commits on the branch, oldest first:

| Commit | Subject |
|---|---|
| `c19a2575fa` | per-section reasoning sampling overrides (upstream PR #25709) |
| `b4c6bbbb77` | reasoning override integration follow-ups |
| `6dd7b2535b` | adaptive MTP draft depth, `draft-mtp-adaptive` (upstream PR #27210) |
| `1816b746c7` | opt-in automatic disk prompt/KV cache, `--slot-save-auto` (upstream PR #24004) |
| `743806bf14` | `test-arg-parser` stderr capture fix |
| `7e27339b58` | `draft-mtp-adaptive` hardening |

Upstream status at the time of integration: #25709 open, #27210 open and under review, #24004
closed without being merged. None of the three is a merged upstream commit, so none of them can
be picked up by a future base bump. They will have to be carried, and if any of them is
eventually merged upstream in a different shape, the fork's copy must be replaced rather than
merged on top.

## 2. Per PR

### 2.1 PR #25709, per-section reasoning sampling overrides

Upstream status: open. The PR adds a second, complete sampling parameter set that applies only
inside a model's reasoning block. It is exposed as `--reasoning-*` CLI flags and `reasoning_*`
server request fields, and it is implemented by wrapping the individual samplers of the one
existing chain in section-aware wrappers that read the current reasoning state before every
apply. The PR also adds `llama_sampler_xtc_set`, so a single xtc sampler object can be
reconfigured in place at the section boundary instead of being rebuilt, and fixes the penalties
sampler's clone dropping `token_count`.

Integration method: squash of the net diff against the v0.4.6 base, not a merge of the PR
branch. The PR is a large refactor of `common_sampler_init` into helper functions, and BeeLlama
has substantial local content in exactly that function. A merge would have produced conflicts
whose resolution was indistinguishable from rewriting the function anyway, and the squash makes
the fork-side decisions reviewable as one diff rather than as conflict resolutions.

BeeLlama-specific adaptations. The whole of BeeLlama's `common_sampler_init` content was carried
into the PR's new helper decomposition rather than dropped: the finite-value validation of
`penalty_repeat`, `penalty_freq` and `penalty_present`; the prefill hardening that checks
`!piece.empty()` and casts to `unsigned char` before `isspace`; the grammar-prefill `try`/`catch`
with its `LOG_ERR` diagnostic; the `skip_grammar_prefill` path for
`COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT` with reasoning tags; the `.vocab` member and its clone; and
the adaptive-p sampler type in the rebuilt chain builder. All of these live in
`common/sampling.cpp`. The reasoning-budget object's creation condition was widened rather than
replaced, so it is now created for any of `grammar_lazy`, an output-format grammar, a
non-negative `reasoning_budget_tokens`, `reasoning_budget_tracking`, `reasoning_control`, or the
PR's new `reasoning_sampling`. The backend-sampling disable warning was widened the same way,
which matters because the section samplers declare no backend entry points: without that gate a
server with backend sampling enabled would have bypassed the section logic entirely.

The EOG repair path is the one place where the fork had to change the PR's shape. BeeLlama's
`common_sampler_force_reasoning_end_on_eog` re-applies the chain after
`common_reasoning_budget_force_end` has flipped the reasoning state. Applying the raw chain there
would have read stale section flags, because the flags are set inside the new
`common_sampler_apply_chain` helper (`common/sampling.cpp:1089`) immediately before each apply.
All three chain-apply sites in `common_sampler_sample` now go through that helper.

`tests/test-chat.cpp` and `tests/test-reasoning-overrides.cpp` were adapted to the fork's APIs:
`eval_llama_cmpl_schema` takes no `n_ctx_slot` parameter here, `common_get_model_or_exit` is the
fork's name, and `reasoning_budget_end` is a `std::vector<llama_tokens>` rather than a flat token
vector.

Deliberate drops. The PR's `PLAN.md` was not taken; it is upstream review scaffolding, not code.
The PR's `-1` to `n_ctx_slot` resolution for the two reasoning last-n parameters was dropped,
because the `-1` convention does not exist in this tree at all: BeeLlama's own
`--repeat-last-n` and `--dry-penalty-last-n` reject negative values and nothing anywhere resolves
`penalty_last_n == -1`. Keeping the resolution would have reintroduced a convention the fork
removed. The schema hard limits for the two reasoning fields were correspondingly declared as
`0..INT32_MAX` in `tools/server/server-schema.cpp` rather than the PR's `-1..INT32_MAX`. One PR
hunk, the `llama_sampler_penalties_clone` `token_count` fix, is absent because the fork already
had it.

Follow-up fixes, commit `b4c6bbbb77`. Review found three defects in the first integration.
`tools/completion/completion.cpp` did not compile: the PR hunk was copied verbatim from upstream,
which has a singular `common_chat_params::thinking_end_tag`, while this tree has a plural
`thinking_end_tags` vector and a `reasoning_budget_end` that is a vector of token vectors. The
completion tool now tokenizes every end tag into the vector and uses the first one as the forcing
sequence, matching what `tools/server/server-schema.cpp` already did. The CLI accepted `-1` for
`--reasoning-repeat-last-n` and `--reasoning-dry-penalty-last-n` while the server threw on the
same value; both now reject negative values, which also removes a silent failure mode where `-1`
was clamped to zero inside the penalties sampler and disabled the penalty instead of widening it.
And `reasoning_dry_penalty_last_n` defaulted to the upstream `-1`, outside its own advertised
schema range; it now defaults to `64` like its non-reasoning sibling.

### 2.2 PR #27210, adaptive MTP draft depth

Upstream status: open and under review. The PR adds a `draft-mtp-adaptive` speculative type that
runs the existing MTP drafting with a per-sequence draft depth controlled by a hysteresis
controller between `--spec-draft-n-min-adaptive` and `--spec-draft-n-max`, instead of a fixed
depth. It also adds `common_speculative_accept_partial`, so a replayed or partially accepted
prefix feeds the controller rather than being lost.

Integration method: squash of the net diff. The PR touches fifteen files, all of which the fork
also touches, and the value of a merge commit here would have been purely historical.

BeeLlama-specific adaptations. The new type is registered in
`common_speculative_type_owns_draft_context` (`common/speculative.cpp:2783`), which is a
fork-only function. That single registration is what makes KVarN draft caches work with
`draft-mtp-adaptive` and what makes `common_validate_draft_kvarn_mode` enforce the "exactly one
owner" rule against a `draft-mtp,draft-mtp-adaptive` combination. The PR converted six raw
`std::find(..., DRAFT_MTP)` lookups into a `common_params_speculative::has_mtp()` helper; the
fork has a seventh, the `-fit` draft-context pairing in `common/common.cpp`, which was converted
too, so `-fit` context sizing is correct for the new type. `docs/beellama-args.md` gained a row
and a note describing the interaction with the DFlash1 adaptive draft-max controller.

Deliberate drops. The PR's `src/models/delta-net-base.cpp` hunk was not taken. It narrows
`build_conv_state` to the snapshot slots reachable within the current batch, which is a valid
optimization upstream, but `src/llama-memory-recurrent.cpp` in this fork advertises
`suffix_rollback_tokens = n_rs_seq` and the server gates rollbacks on that bound alone,
independent of the last batch's token count. Slots outside the batch can therefore still be
rolled back into, and narrowing the conv-state build would have been unsound. Dropping an
optimization is the safe direction; the commit message records the reasoning so a future rebase
does not silently re-apply it.

Follow-up fixes, commit `7e27339b58`. The new per-sequence arrays made `begin()` index without
the bounds check every other entry point in the same class uses, which turned a previously
harmless out-of-range `seq_id` into undefined behaviour; `begin()` now guards like its siblings.
The adaptive range check threw out of `common_speculative_init` when the model's MTP layer count
capped `n_max` below the default `n_min_adaptive` of 3, which aborted server startup for any
model with exactly two MTP layers under default flags; the floor is now clamped to the
model-derived cap with a warning (`common/speculative.cpp:1516`) instead of aborting. The
header-only `common/speculative-adaptive.h` was added to the common target's source list, and the
docs now say plainly that the stock defaults leave the controller no room to climb, since
`n_max` and `n_min_adaptive` are both 3 and the user has to raise `--spec-draft-n-max` before
adaptivity does anything at all.

### 2.3 PR #24004, automatic disk prompt and KV cache

Upstream status: closed without being merged. The PR adds an opt-in `--slot-save-auto` mode in
which the server writes a slot's prompt and KV state to disk when the slot goes idle and restores
it on a later matching request, with `--slot-save-block`, `--slot-save-max-count` and
`--slot-save-max-mb` controlling granularity and the on-disk budget. Because upstream closed it,
the PR was treated as a design rather than as a patch to apply, and the port is a squash of a
reimplementation on the v0.4.6 base.

BeeLlama-specific adaptations, all in `tools/server/server-context.cpp` unless noted.

The snapshot fingerprint is the largest deviation. Upstream compares an explicit field list.
Restoring a KV blob into a context whose geometry differs is silent corruption, and this fork has
far more geometry than upstream does, so the fingerprint gained two 64-bit hashes rather than
roughly twenty-five more explicit fields. `auto_bee_kv_hash()` covers the standard cache types,
the whole KVarN descriptor including bit widths, group, Sinkhorn iterations and sink tokens, the
KVarN bit intents, the KV precision tail tokens and type, `swa_full`, `kv_unified` and
`kv_unified_per_slot`, and `n_batch`, `n_ubatch` and `n_parallel`, because
`llama_kv_cache_kvarn` derives its tail, stage and per-stream group counts from exactly those.
`auto_bee_dft_hash()` covers the speculative types, the draft model path, the draft cache types
and KVarN descriptor, and `draft_owns_state`; a draft-owning slot never writes a snapshot, but a
draft mode still changes the target context, so the draft configuration is identity rather than
metadata. `operator==` is an exact all-field comparison, so a hash is semantically identical to
explicit fields while removing about two hundred lines of hand-written serialization and the
read/write field-order mismatch that goes with it. `SLOT_META_VERSION` was bumped to 2
(`tools/server/server-context.cpp:640`), so a snapshot written by an upstream-layout build is
rejected by the version check rather than misparsed. `fp_kv_full` was repurposed from upstream's
"is the memory FULL-capable" boolean to the full seq-rm capability enum
(`tools/server/server-context.cpp:3630`), because KVarN reports a third class, `RS`, and
collapsing it into "not FULL" would make a KVarN snapshot look interchangeable with a standard
one.

Block alignment. `llama_kv_cache_kvarn::get_seq_rm_capability()` clamps suffix rollback to the
128-token KVarN group, so `common_context_can_seq_rm()` classifies a KVarN target as `RS`, and
the server's own `prompt_reuse_alignment()` is the KVarN group when KVarN is active and 1
otherwise. `--slot-save-block` must therefore be a multiple of the group; this is enforced at
parse time in `common/arg.cpp:1144` so a bad combination fails fast, with a defensive warning at
server init. The default block of 256 is two groups, so the default configuration is already
legal.

Restore policy. Upstream special-cases `FULL` memories and lets everything else clamp a restore
to a block boundary. The port inverts that: `auto_restore_into_slot` special-cases `PART` and
makes both `FULL` and `RS` require the whole snapshot to be a verified prefix of the request. A
request that diverges inside a snapshot is refused and cold-prefills rather than issuing a
rollback KVarN would reject, which the server treats as a hard error rather than a fallback. A
snapshot is still indexed at every block boundary, so a longer request that extends a shorter
snapshot reuses it fully; the block size controls index granularity and the margin gate, not what
can be restored.

Owned draft caches are gated off entirely. `auto_slot_excluded()`
(`tools/server/server-context.cpp:1784`) returns true for any slot with `draft_owns_state` and
emits one warning for the process lifetime. It is called first in `auto_save_slot_if_useful` and
again in the auto-restore gate on the prefill path. The reason is that
`llama_state_seq_save_file` persists the target sequence only; restoring it would leave the
slot's draft context holding a stale or empty sequence for the same slot id, while the checkpoint
restore transaction and the speculative rollback logic both assume the two move in lockstep.
Doing this properly needs a second state file, a second fingerprint and an atomic five-file
publish. The gate is per slot rather than server-wide, so a server with mixed slots behaves
correctly, and it costs nothing for n-gram drafting or for non-speculative servers.

Logits capture. Upstream captures the last decoded token's logits inline at its single sampling
site. This fork has two, so the capture was factored into `slot_capture_logits`
(`tools/server/server-context.cpp:3496`), keeping upstream's two gates so that KVarN servers and
servers without `--slot-save-path` pay nothing. On the speculative path the capture must be the
distribution that produced the last accepted token, not the last drafted one: the capture is
invalidated and `spec_logits_idx` recorded immediately after the sampler returns
(`tools/server/server-context.cpp:6166`), before `spec_i_batch` is cleared, so that a checkpoint
rollback a few lines later leaves no stale capture behind, and the actual capture happens after
the accepted prefix has been written back into the slot's token list
(`tools/server/server-context.cpp:6283`).

Replayed tokens go through accept-with-info. In the restore-continue fast path the token produced
by `common_sampler_sample_from_logits` is fed to `common_sampler_accept_with_info` and then to
the reasoning loop guard, not to the plain accept upstream uses. Without that, the reasoning
budget and the loop guard would silently fail to advance for the replayed token. That function
also mirrors the fork's full-logits sampling path including the EOG repair step, and it was
merged with PR #25709's change: its three chain applies go through `common_sampler_apply_chain`
(`common/sampling.cpp:1412` onward), so per-section reasoning overrides are honoured for replayed
tokens too.

Smaller deviations, recorded so they are not mistaken for accidents. The auto path writes and
reads `server_tokens::serialize()` and `deserialize()` rather than bare token ids, so manually
and automatically saved snapshots are mutually readable and `server_tokens::validate` still runs
on every restore; the `.meta` sidecar keeps raw token ids, since it is the hashing key rather
than the payload. `do_slot_restore` uses the fork's two-pass load and reports failures through an
out-parameter, so the manual endpoint keeps its detailed error text. The checkpoint rebuild after
a restore is guarded on `slot.task != nullptr`, because the manual restore endpoint runs on an
idle task-less slot where upstream would have null-dereferenced on any recurrent model, and its
condition was widened from "is FULL" to "is not PART" so KVarN gets the checkpoint too. Auto-save
is wired into the fork's inlined idle flush and into `get_available_slot` before the RAM prompt
cache is disabled, with the two sites made mutually exclusive so nothing is saved twice.
Prompt-cache telemetry reports `disk` as the source for a successful auto-restore so the existing
attribution is not lost. Upstream's `get_timings` divide-by-zero fix was not ported because the
fork already guards both divisors. Restore-continue is deliberately `FULL`-only: for `RS` and
`PART` a no-suffix regenerate needs a one-token rollback that the fork's live rollback plan
already performs natively, so the consequence is that the `.logits` sidecar only ever exists for
recurrent and hybrid targets.

### 2.4 The test fix

Commit `743806bf14` is not part of any PR. `capture_stderr` in `tests/test-arg-parser.cpp`
asserted that `dup2` returns 0, but POSIX `dup2` returns the new descriptor, so the test aborted
at its first capture and every case after that point never ran. It now compares against the saved
descriptor. The Windows `_dup2` branch does return 0 on success and is unchanged. This is worth
knowing when reading older green test runs on this file: before this commit, a passing
`test-arg-parser` proved much less than it appeared to.

## 3. Interaction with KVarN and owned draft caches

The three changes sit at very different distances from the cache.

`draft-mtp-adaptive` is registered as an owned-draft-context mode. That is the whole of its cache
interaction, and it is deliberate: the type owns its draft KV context, so the KVarN draft cache
presets selected with `--spec-draft-type-k` and `--spec-draft-type-v` apply to it exactly as they
do to plain `draft-mtp`, and `common_validate_draft_kvarn_mode` will reject a configuration that
pairs it with another owning type. The adaptive depth itself changes only how many tokens are
drafted per round, which the draft cache already handles for every other variable-depth mode. The
controller is not serialized in the speculative state envelope, so a checkpoint restore into a
different slot keeps whatever depth that slot had; this is self-correcting within a few verifies
and reset at the next generation.

`--slot-save-auto` is the change that touches the cache directly, and its three KVarN-facing
mechanisms are the fingerprint, the gates and the block alignment described above. In short: the
fingerprint refuses a restore whose KV geometry differs, before paying for a multi-gigabyte read;
the block size is constrained to a multiple of the KVarN group because that is the rollback
granularity the cache advertises; `RS` and `FULL` memories restore whole verified snapshots only,
so no restore can be followed by a rollback the cache would reject; and any slot that owns draft
state is excluded from both saving and restoring. The underlying round trip is supported, not
improvised: `llama_state_seq_save_file` and `llama_state_seq_load_file` with zero flags reach
`llama_kv_cache_kvarn::state_write` and `state_read`, which validate magic, state version, cache
type, layer count, stream count, state kind and stage and tail group counts, throwing on any
mismatch. The wrappers turn a throw into a zero return, so a geometry mismatch that slips past
the fingerprint still degrades to "restore failed" rather than to corrupt state.

The reasoning sampling overrides have no KV cache interaction at all. They change which sampler
parameters apply to a token, not what is stored or how it is stored, and the only `src/` change
in that PR is host-side sampler code.

## 4. What was verified here, and what was not

Verified: a CPU-only Release build of `llama-server` at branch HEAD, and the following ctest
targets, all passing.

| Target | Covers |
|---|---|
| `test-sampling` | sampler behaviour including the xtc changes from #25709 |
| `test-chat` | the reasoning effort caps and per-request reasoning sampling cases |
| `test-chat-peg-parser` | chat parsing, regression only |
| `test-chat-auto-parser` | chat parsing, regression only |
| `test-chat-template` | chat templates, regression only |
| `test-download-model` | model resolution, regression only |
| `test-arg-parser` | the new `--reasoning-*` and `--spec-draft-n-min-adaptive` parsing, with the `dup2` fix in place |
| `test-reasoning-overrides` | the #25709 section switching, against the tinyllamas fixture |
| `test-speculative-adaptive` | the #27210 depth controller state machine in isolation |
| `test-adaptive-dm` | the pre-existing DFlash1 profit controller, regression only |
| `test-server-prompt-checkpoint` | server prompt checkpointing, regression only |

Every PR hunk was also reconciled file by file against the upstream PR diff, and every touched
source file was syntax-checked with the compiler. The reviews recorded in the working notes are
read-only analyses of the integration, not test runs.

Not verified. The integration environment has no CUDA toolchain and no model weights, so nothing
below has been observed running.

The residual risks worth carrying into a GPU run, collected from the three reviews and the port
notes:

From #25709: whether the section parameters flip on exactly the right token at the tag boundary,
since the chain is applied before the token is accepted and the token that completes a tag is
therefore sampled under the old section; whether the reasoning-section parameters being used
during budget forcing is genuinely unobservable, as the analysis says it is, given that the
budget pins the token anyway; the observable effect of switching mid-generation into a penalty or
DRY sampler with a shorter last-n window; and whether the xtc RNG stream, which is continuous
across sections because the one sampler object is mutated in place, matches a seeded reference
run.

From #27210: the controller's behaviour under a real acceptance trace, since the unit test covers
the state machine only; and the startup path on a model with exactly two MTP layers, which is
what motivated the floor-clamping fix and which no model here can exercise. Note also that the
serialized speculative state envelope embeds the raw enum ordinal and inserting the new type
renumbered the types after it. This is harmless today because slot checkpoints and the prompt
cache are RAM-only within one process, but if a persisted or on-the-wire form of that envelope is
ever added, the envelope version must be bumped or the type tag decoupled from the ordinal.

From #24004: that a KVarN state blob written by one process actually loads into an identically
configured second process, which is an on-disk format claim and the single most important thing
to test; that the restore-continue fast path produces the same first token a live decode would,
which depends on the sampler replaying its accept history over the restored prompt; that the
strict whole-snapshot-only restore policy for `RS` and `FULL` is not so conservative in practice
that it refuses reuse the live rollback plan could have handled, which wants a benchmark rather
than a pass/fail; that the speculative capture index is the right one against real logits; and
the cross-process eviction, directory mtime refresh and atomic three-file publish, which were
carried over from upstream unchanged and never exercised. No new tests were added for this PR and
the `tools/server/tests` suite was not run, because it needs a model.

## 5. Runtime verification checklist

Run these on a machine with the GPU and the models. Each item names what it is actually proving,
because several of them look like smoke tests and are not.

Build with `scripts/build-server-cuda.sh` and read `docs/build-server-cuda.md` first. The default
run compiles the 15 balanced KVarN fast-decode pairs and the 50 standard FlashAttention vector
pairs; `FA_ALL_QUANTS=1` expands those to 36 and 169 at roughly two to two and a half times the
CUDA wall clock, and is only worth it for an unbalanced K/V pair outside the default matrix. The
script builds `llama-server` alone.

Confirm the KVarN fast path before measuring anything else. Start the server with
`--cache-type-k kvarn4 --cache-type-v kvarn4 --flash-attn on` under `GGML_KVARN_DEBUG_ROUTES=1`
and generate a few tokens. The startup line must say that the structured KVarN cache type was
enabled rather than that it fell back, and the per-op trace must show `route=decode-split` or
`route=decode-vector` with `fallback=none` during single-token decode. `route=generic-mma` means
the pair was not compiled as a fast-decode instance, which for `kvarn4`/`kvarn4` means the build
is wrong. `route=prompt-generic-mma` during prefill is normal for every pair.

Exercise the reasoning overrides with a real reasoning model and a distinguishable setting, for
example `--reasoning-temp 0` against a high base temperature. The point is not that the flag
parses, which the tests already cover, but to see where the switch happens relative to the
thinking tags and to confirm the off-by-one at the boundary is the intended one.

Exercise the adaptive draft depth with `--spec-type draft-mtp-adaptive --spec-draft-n-max 12` and
`-lv 1`. The default `--spec-draft-n-max` leaves the controller pinned at its floor, so a run with
default flags proves nothing; with the cap raised, the debug log should show the per-sequence
adaptive draft depth line firing and the depth actually climbing on an acceptance-friendly
workload. A partial-acceptance-heavy workload additionally exercises the `accept_partial` path.

Exercise cross-process disk restore, which is the claim with the least evidence behind it. Run a
server with `--slot-save-auto --slot-save-path <dir> --cache-type-k kvarn4`, send a long prompt,
let the slot go idle so the snapshot is written, stop the process, start a second server with an
identical configuration and the same path, and send the same prompt again. The second process
must report a disk restore rather than a cold prefill, and the generated continuation must be
sane. Then repeat with one KVarN parameter changed, for example the group or a bit width, and
confirm the fingerprint refuses the restore instead of loading it. Both directions matter: the
first proves the format round-trips, the second proves the guard works.

Run the `tools/server/tests` pytest suite against a real model. It was not run here at all, and
it is the only end-to-end coverage the server side of #24004 has.
