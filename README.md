Copyright © 2026 Lewis James Seals Jr. All rights reserved.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT.

# Neural Compressor — distillation harness

**This is a distillation harness, not evidence that a neural compressor is better
than a classical one.** The model is trained to reproduce the gain curve and the
four control parameters of a classical feed-forward compressor with a soft knee
and attack/release. The teacher is an upper bound on what this objective can
express: there is no perceptual loss, no loudness (LUFS) term, no adversarial
term, and no listening test. If you want to argue that a neural compressor sounds
better, this objective cannot express that claim and this repo does not test it.

What it does test: whether a small CNN can learn a compressor's *behaviour*
(its gain-reduction trajectory as a function of program material) from audio
alone, well enough to predict it on held-out material.

## Quick start

    python -m pip install -e ".[dev]"
    python -m training.train --dry-run
    python -m training.train --out-dir checkpoints

The defaults (`n_samples=4096`, `epochs=20`, 1.5 s clips) are a **smoke test**,
not a training run — roughly 3.5 hours of synthetic audio and ~10k optimizer
steps. See §4 for what a real run needs.

## Repository layout

    model/blocks.py     the network; forward() -> (p_pred, g_pred, aux)
    model/losses.py     the objective; the only place `total` is computed
    training/dataset.py synthetic program material + the classical teacher
    training/train.py   harness: held-out validation, AMP, resume, reporting
    tests/              model, loss, dataset and end-to-end tests

## Evaluation protocol

`best.pt` is selected on **validation curve loss** over a disjoint synthetic set
(`seed + 7919`, `--n-val` samples). The old script selected on the first 64
*training* samples; that number was not a generalisation metric.

## Band Head Rationale

The band head present in earlier drafts is intentionally absent: the teacher is a
single-band compressor, so every band's target gain is identical and a per-band
head has no target to learn.

## What a Real Run Needs

- Hours of real program material instead of synthetic generation
- $\geq 50\text{k}$ training steps
- A perceptual or LUFS alignment term
- A blind listening test for evaluation
