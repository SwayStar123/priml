# SpeedrunDiT baseline

`priml.baselines.srdit.experiments.exp000` implements the linked
`invae-sprint-rms-rope-valres-cfm-muon-layerwisescaling` branch of
[SpeedrunDiT](https://github.com/SwayStar123/REG/tree/invae-sprint-rms-rope-valres-cfm-muon-layerwisescaling).
The model uses SiT-B/1 (32-channel, 16×16 INVAE latents), REG's diffused
DINOv2 CLS token and intermediate projection loss, SPRINT's 2/8/2 dense/sparse/dense
split with 75% token routing, 2D RoPE, RMSNorm, QK normalization, first-block
value residuals, and a 2→6 layerwise MLP ratio. The objective combines latent
velocity MSE, CLS velocity MSE (0.03), DINO alignment (0.5), and contrastive
flow matching (0.05). Training shifts times using the full latent dimension,
uses bf16 autocast and EMA (0.9999), and routes hidden matrices to Muon while
AdamW trains the embeddings, conditioning, CLS input projection, and output
layers. The REG projection MLP matrices go to Muon under the linked branch's
parameter-name rule.

## Data

The original ImageNet source in priml reads raw extracted JPEGs. This recipe
requires *paired* preprocessed images and sampled INVAE latents, so its source
indexes those pairs with a map-style PyTorch dataset. A `DataLoader` batches
them with the reference's random sampling order on one device. Distributed
training uses a sampler with disjoint per-device partitions. It expects:

```text
<base_dir>/datasets/srdit/
  images/00000/img00000000.png
  vae-in/00000/img00000000.npy
  vae-in/dataset.json
```

`dataset.json` has the reference format
`{"labels": [["00000/img00000000.npy", 0], ...]}`. Relative stems under
`images/` and `vae-in/` can match, or use the reference's
`img00000000.png` / `img-latents-00000000.npy` naming. The JSON key must name
the latent file. INVAE arrays contain an unscaled sampled
latent with shape `[1, 32, 16, 16]` at 256 resolution, or `[1, 32, 32, 32]`
at 512 resolution. The train step applies the reference scale `0.3099`.

Use the [reference preprocessed dataset](https://github.com/SwayStar123/SpeedrunDiT#dataset),
or create it from priml's existing extracted ImageNet source:

```bash
python -m priml.baselines.srdit.scripts.prepare_data \
  --source /datasets/imagenet \
  --output /opt/scratch/datasets/srdit \
  --resolution 256
```

The preparer and decoder accept a local `--checkpoint` or download
`REPA-E/e2e-invae` with the optional `priml[hub]` dependencies. DINOv2 teacher
weights load through `torch.hub` on the first training run. The vendored INVAE
architecture and MIT notice are in `priml/model/third_party/`.

## Train

The published effective batch size is 256: 32 samples per device on eight
devices. Launch the distributed recipe with priml's standard launcher:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 -m priml \
  priml.baselines.srdit.experiments.exp000
```

For a five-update single-device check, use
`python -m priml priml.baselines.srdit.experiments.exp_smoke`.
Set `--override base_dir=/your/storage/root` to change the dataset and run root.
The recipe uses priml's checkpoint and resume mechanism. It deliberately skips
validation during training, as the reference evaluates generated images in a
separate workflow.

`sample_latents` in `sampling.py` implements the reference Euler-Maruyama SDE
with optional classifier-free guidance. Pass the returned image latents to
`decode_latents` in `vae.py`. `path_drop_guidance=True` is for qualitative
images; use the default `False` for metric runs. The sampling API returns
latents and CLS samples, so callers can choose their own image grid and metrics.

## Reference differences

The priml run uses its native checkpoint and EMA implementations. Its paired
dataset uses PyTorch data loading, and its Muon arithmetic matches the linked
branch. It reads processed ImageNet/INVAE pairs directly and
computes DINO features online, as the branch does. Quantitative FID/KDD
evaluation and the reference's W&B image logging are separate from this
training recipe; no published score is claimed for the priml implementation.
The CPU parity check uses fixed model inputs and compares forward outputs and
parameter gradients. The smoke golden records five optimizer updates. A full
eight-GPU training run has not been verified.
