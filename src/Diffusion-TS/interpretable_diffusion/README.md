# Interpretable Diffusion

PyTorch components of [Diffusion-TS](../README.md): the diffusion process, a Transformer that decomposes time series, and classifier guidance.

## Architecture

`Diffusion_TS` takes windows of shape `(B,L,F)` and a diffusion timestep `t`. `Transformer` embeds features using convolution, adds positional encoding, and processes them through an encoder–decoder. The timestep controls `AdaLayerNorm` in the attention blocks.

| Module | Main Classes/Functions | Role |
| --- | --- | --- |
| [gaussian_diffusion.py](gaussian_diffusion.py) | `Diffusion_TS`, `linear_beta_schedule`, `cosine_beta_schedule` | Forward diffusion, reconstruction loss, sampling, and conditioning |
| [transformer_custom.py](transformer_custom.py) | `Transformer`, `Encoder`, `Decoder` | Encoder–decoder denoiser; returns trend and seasonality with a residual |
| Same file | `TrendBlock`, `FourierLayer` | Polynomial trend of degrees 1–3; selects high-amplitude frequencies to reconstruct oscillations |
| [model_utils.py](model_utils.py) | `Conv_MLP`, `LearnablePositionalEncoding`, `AdaLayerNorm`, `extract` | Embedding, timestep conditioning, and broadcasting diffusion coefficients |
| Same file | `cond_fn` | Computes the gradient of the target class log-probability to guide sampling |
| [classifier.py](classifier.py) | `Classifier` | Timestep-conditioned encoder and attention pooling to produce class logits |

`DecoderBlock` splits the representation into trend and Fourier branches; the decoder sums contributions from multiple blocks. `Transformer.forward(..., return_res=False)` returns two `(B,L,F)` tensors that are summed to form the reconstruction. With `return_res=True`, it returns three tensors: trend, seasonality, and residual. `MovingBlock` and `SeasonBlock` are present in the source but are not selected in the default decoder.

## Training loss

`forward(x, **kwargs)` randomly samples `t` and calls `_train_loss`. `q_sample` adds noise according to the beta schedule; `output` sums the two Transformer outputs into a prediction of the clean sequence. The default target is `x_start`. `loss_type` selects L1/L2; `use_ff` adds an FFT error term; `loss_weight` reweights the loss by timestep.

`forward` only returns a scalar loss. The optimizer loop, learning rate schedule, and checkpointing are implemented externally, as in the [API example](../README.md).

## Sampling and conditioning

| API | Key Inputs | Output |
| --- | --- | --- |
| `sample(shape)` / `fast_sample(shape)` | `shape=(B,L,F)` | Sequences generated from noise; full/reduced-step paths |
| `generate_mts(batch_size, model_kwargs=None, cond_fn=None)` | Batch size; callback when guidance is used | Tensor `(B,L,F)` |
| `sample_infill(shape, target, partial_mask, ...)` | Target `(B,L,F)` and a boolean mask of the same shape | Samples that preserve observed entries in the final output |
| `fast_sample_infill(shape, target, sampling_timesteps, ...)` | As above, with a reduced number of steps | Samples with missing entries filled in |
| `return_components(x, t)` | Sequence and integer timestep | `(trend, season, residual, noisy_x)` |

For infilling, **`partial_mask=True` marks observed entries**, while `False` marks entries to generate. `model_kwargs` must be a dictionary providing `coef` and `learning_rate` for `langevin_fn`; it may also include `coef_` for perturbation noise. The default `None` is insufficient for this path because the code unpacks `**model_kwargs`. The observation mask must contain valid entries so that the loss does not compute a mean over an empty set.

Classifier guidance uses `model_utils.cond_fn(x,t,classifier,y,classifier_scale)`, where `y` is the target class label. The classifier must be trained separately to handle noisy inputs/timesteps; no classifier trainer is provided here. The module does not automatically convert driver metadata into conditions.

## Usage Requirements

- Keep `L = seq_length` and `F = feature_size` fixed. Positional encoding is added at its full length rather than sliced to match the input. `d_model` must be passed explicitly; the default `None` cannot construct the embedding layers.
- `FullAttention` and `CrossAttention` accept a `mask` argument but do not apply it to attention scores. Therefore, `padding_masks` does not guarantee that padding is ignored; prefer equal-length windows containing finite numeric values. The loss does not exclude padded entries either.
- `FourierLayer.topk_freq` computes the `log` of the number of frequency bins after removing boundary bins. Very short sequences may have no valid bins; do not assume every `seq_length` will work.
- For `Classifier`, set `max_len=seq_length`; `GroupNorm32(8, seq_length)` requires the length to be divisible by 8, and `num_head_channels` must match the attention pooling dimensions. The `__main__` block is a CUDA forward-pass example, not a training CLI.
- The `linear` schedule multiplies beta by `1000/timesteps`; drastically reducing the number of training steps can produce invalid beta values. To reduce generation cost, distinguish `sampling_timesteps` from `timesteps`.
- Importing `transformer_custom`/`model_utils` requires this directory to be on the import path. The module does not automatically save artifacts or normalize data.

## Further Reading

[Diffusion-TS README](../README.md) · [Paper](https://arxiv.org/abs/2403.01742) · [Authors' Repository](https://github.com/Y-debug-sys/Diffusion-TS)

The docstring for `AttentionPool2d` in `classifier.py` attributes it to [OpenAI's CLIP](https://github.com/openai/CLIP/blob/main/clip/model.py); the class name is retained even though the code here processes sequence tensors.
