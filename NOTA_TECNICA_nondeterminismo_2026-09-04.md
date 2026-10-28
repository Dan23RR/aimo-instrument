> Nota tecnica da allegare, se Michal la chiede. Non è la mail: la mail è la versione breve
> in `MAIL_organizzatori_2026-09-04.md`.

# Re-execution noise in the AIMO Interpretability evaluation: what we measured and what we did not

**Daniel Culotta** (Codabench `dan23rr`). Working note, 4 September 2026.
Every figure below is recomputed from our own submission logs and extraction files. Where a
claim is an inference rather than a measurement, it is labelled as such.

## Summary

The nondeterminism we reported is not sampling. We have not identified its mechanism. What we
can show is that batch composition changes the extracted numbers for identical inputs, both in
a controlled test on our own hardware and in one paired case inside your container.

**The nondeterminism we reported is not sampling.** Our four byte-identical submissions
(909976, 909994, 910002, 910013, scoring 11, 9, 11, 11 out of 17) all ran under greedy
decoding, because your own package requires it: `uncertainty_profile/config.py` defaults
`do_sample` to False and its `validate()` raises "uncertainty profiling requires
deterministic greedy decoding" on anything else. So the 2-item spread we sent you came from a
fully greedy path. Sampling is an additional source of variance for teams that use it, but it
is not the source of what we measured, and removing it would not remove what we measured.

**What we have not done is identify the mechanism.** Our technical report says so in as many
words, and I would rather repeat it here than let this mail imply more: we measured the
magnitude and the structure of the drift, not its root cause. What follows is the evidence,
with its limits attached.

**Batch composition changes the numbers for identical inputs.** Two observations, one
controlled and one from inside your container.

The controlled one is ours and it is narrow: one model (DeepSeek-R1-0528-Qwen3-8B), 40
problems, a single forward pass over the prompt with no generation, one hidden layer (29 of
36), on a rented A100 rather than in your container. We computed four scalar features from the
last prompt token's activation (cosine distances between the clean prompt and two mechanical
perturbations of it, plus two combinations) twice, varying exactly what varies in production:
batch 1 with no padding, against batch 8 with left padding and different neighbours. Mean
relative difference 1.1 to 1.5 percent, ICC(2,1) across the 40 problems between 0.997 and
0.999. That bounds the prompt-side arithmetic on our hardware. It says nothing about your
GPUs, and I would not want it read as if it did.

The second observation is inside your container and I find it more telling. Our entry fits the
gpt-oss decision threshold at runtime on the same 10 public labelled `val-sample` items,
shipped in the bundle. Same items, same model, same code: the fitted threshold came out
**0.563 in the warmup and 0.521 in the main phase**. Those calibration items are prepended to
the evaluation batch, so between the two phases the main thing that changed for them was the
batch they were embedded in. Some of the gap may be the container itself changing between
phases, which I cannot rule out from outside. But it is a paired case where identical inputs
produced different extracted features on your hardware.

**The score then moves only when drift crosses a decision threshold.** That step is measured.
In run 4 every regressor value we logged differed from the other three runs, yet the splits
coincided with run 1 and the score came out the same 11/17. So the visible score is a
threshold function of an underlying continuous drift, which is why it moves in whole items and
can look bimodal when it is not.

**Where I would look next, offered as a hypothesis and not a finding:** the decode chain. Your
artifact sets `max_new_tokens` to 4096, and in our own greedy extraction on 164 public
problems the median generated length was exactly the 4096 cap, with 90 of 164 items hitting
it. A single argmax flip early in a chain that long changes everything after it. We never
logged tokens or logits, by choice, so we have not observed such a flip, and kernel selection
or batch scheduling would explain the same data equally well.

**One structural detail, with the confound stated.** Across all four warmup runs the logged
score range was identical to three decimals for `google/gemma-3-27b-it` ([0.425, 0.789]) and
`openai/gpt-oss-120b` ([0.340, 0.902]), while `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` produced
three distinct ranges and `allenai/Olmo-3-7B-Instruct` two. Our logs record only the per-model
minimum and maximum, so for the two-item batches that is the whole batch, while for gpt-oss it
is two order statistics out of nine: read it as the extremes not moving.

Item count does not explain the split, since gemma and DeepSeek both had exactly two items and
only DeepSeek moved. Model size would, but I cannot claim it: your harness calls the models in
ascending size order (Olmo 14.2, DeepSeek 15.8, gemma 51.8, gpt-oss 61.4 GiB), so the two
drifting models are also the two called first, on a cold CUDA context and an empty allocator.
With four models those are the same variable. **Permuting the call order in runs you are
already doing would separate them at no extra compute**, and that is the cheapest experiment I
can suggest.

**On the number of repeats, an honest note against my own recommendation.** Our SD across the
four runs is 1.0 item on 17. Averaging k runs leaves a standard error of 1.0/sqrt(k), so k=3
gives 0.58 items and k=8 gives 0.35. For a one-item gap between two teams to be two standard
errors of the difference you need k around 8, not 3. With only four samples the 95 percent
interval on that SD is roughly [0.6, 3.7] items, so the required k is itself poorly determined.
Three runs will make the published number much more stable, which is worth doing, but on its
own it will not decide a ranking at one-item resolution.

I should also declare the obvious conflict of interest, in the direction that goes against us:
averaging lowers our own published figure. Our 0.647 is the maximum of those four samples; the
mean is 10.5/17 = 0.618. We would still rather be ranked on the mean.

Which is why I would not pay for three runs of everything. Three designs, in the order I would
try them:

1. **Try determinism first, since this is not sampling.** Pin batch shapes per submission, set
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialises, and use
   `torch.use_deterministic_algorithms(True, warn_only=True)` in the harness. The `warn_only`
   matters: in strict mode PyTorch raises a RuntimeError on the first operation with no
   deterministic kernel, which in a harness running arbitrary participant code would fail
   legitimate entries rather than merely slow them. For what it is worth, the shipped pipeline
   today sets no seed and calls neither. These flags only buy reproducibility at a fixed GPU
   model, fixed library versions and fixed input shapes, so this is something to verify rather
   than assume.

2. **Measure the instrument once, not once per submission.** Run a reference bundle k times in
   a quiet window and publish, per phase, "re-execution SD of this reference bundle: X items".
   That is k runs in total, it gives every participant a scale for reading gaps, and it tells
   you whether step 1 worked. One caveat from our own data: this SD is a property of the bundle
   as much as of the harness. An entry whose decisions sit far from its thresholds has SD
   exactly zero, and much of the warmup field was sitting on constant predictors. Publish it as
   an order of magnitude, not as a universal error bar.

3. **Repeat only where it decides something, and only at the close.** During the phase, keep one
   run per submission as now. At the freeze, re-run only the entries close enough to a prize
   boundary to be in doubt, with eight repeats each, since five would only resolve a 1.3-item
   gap. For m contested entries that is 8m extra runs per task, against the 2N extra runs a
   blanket triple run would cost for N scored entries. You have N and I do not, but for any m
   much smaller than N the saving is large, and it buys exactly the comparisons that assign
   prizes.

The raw per-model score ranges from the four runs are available on request.
