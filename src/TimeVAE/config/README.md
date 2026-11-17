# TimeVAE Configuration

[hyperparameters.yaml](hyperparameters.yaml) contains hyperparameters for the three [VAE](../vae/README.md) variants. The top-level key is the `vae_type` value passed to `instantiate_vae_model`: **`timeVAE`**, **`vae_dense`**, **`vae_conv`**; names are case-sensitive.

## Shared Parameters

| Key | Value in YAML | Meaning |
| --- | --- | --- |
| `latent_dim` | `8` | Number of Gaussian latent dimensions |
| `hidden_layer_sizes` | `[50,100,200]` | Number of Dense units or Conv1D filters, depending on the variant |
| `reconstruction_wt` | `3.0` | Weight of the reconstruction loss relative to KL |
| `batch_size` | `16` | Number of sequences per batch in `fit_on_data` |

`sequence_length` and `feature_dim` are inferred from the data `(N,L,F)` rather than declared in YAML. The factory converts them to `seq_len` and `feat_dim` when constructing the model.

## Parameters Specific to `timeVAE`

| Key | Value in YAML | Meaning |
| --- | --- | --- |
| `use_residual_conn` | `true` | Enables the convolutional residual decoder |
| `trend_poly` | `0` | Disables the trend; a positive value specifies the highest polynomial degree |
| `custom_seas` | `null` | Disables the custom seasonal component |

Example seasonality configuration, **not yet verified by execution**, suitable only for data with a one-day timestep:

```yaml
custom_seas:
  - [7, 1]
  - [12, 30]
```

Each pair is `[num_seasons, len_per_season]`: seven positions with one step each create a seven-step cycle; twelve positions with thirty steps each create a 360-step cycle. This is not an actual calendar month/year; choose values according to the data frequency and windowing scheme. A similar example is already included as a comment in the YAML file.

## Loading Configuration

`load_yaml_file(path)` uses `yaml.safe_load`; the pipeline selects the `[vae_type]` group and passes its keys to `instantiate_vae_model`. There is no configuration merging or CLI override mechanism.

Settings defined outside YAML:

- `run_vae_pipeline`: validation ratio `0.1`, shuffling enabled, `max_epochs=1000`.
- `instantiate_vae_model`: seed `123`; each model is compiled with the Adam optimizer.
- `fit_on_data`: EarlyStopping `patience=50`, `min_delta=1e-2`; ReduceLROnPlateau `patience=30`, `factor=0.5`.
- `paths.py`: data, configuration, and output paths; see [pipeline paths](../README.md).

This file is actually located at `src/TimeVAE/config/hyperparameters.yaml`, while `HYPERPARAMETERS_FILE_PATH` currently resolves to `src/src/config/hyperparameters.yaml`. Align the paths before running the pipeline; the API example in the parent README reads the correct path directly. There is no additional `config.yaml` in this directory.

## Further Reading

[TimeVAE and Usage](../README.md) · [Encoder, Decoder, and Loss](../vae/README.md) · [Authors' Repository](https://github.com/abudesai/timeVAE)
