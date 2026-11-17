# VAE Models

Encoders, decoders, and objective functions for the three VAE variants in [TimeVAE](../README.md). All models take windows of shape `(N,L,F)` and produce outputs of the same shape, allowing comparison of the effects of encoder–decoder architecture.

## Variants

| `vae_type` Option | Class | Architecture |
| --- | --- | --- |
| `vae_dense` | `VariationalAutoencoderDense` in [vae_dense_model.py](vae_dense_model.py) | Flatten the window, use Dense encoder/decoder layers, and reshape the output to `(L,F)` |
| `vae_conv` | `VariationalAutoencoderConv` in [vae_conv_model.py](vae_conv_model.py) | Conv1D encoder with stride 2, Conv1DTranspose decoder, and a final Dense layer to match dimensions |
| `timeVAE` | `TimeVAE` in [timevae.py](timevae.py) | Conv1D encoder; decoder sums level/trend/seasonality/residual |

[vae_base.py](vae_base.py) defines `BaseVariationalAutoencoder` and the reparameterization class `Sampling`. This is an abstract base class, not a variant selectable through `vae_type`. [vae_utils.py](vae_utils.py) provides the creation, training, saving/loading, and sampling APIs used by the pipeline.

## Encoder and Objective Function

The encoder returns `z_mean`, `z_log_var`, and a latent sample `z`, each with shape `(N,latent_dim)`. `Sampling.call` uses:

$$
z = \mu + \exp(\tfrac{1}{2}\log\sigma^2)\odot\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I).
$$

`train_step` decodes `z`, computes the reconstruction loss and KL divergence to a standard Gaussian, and applies gradients using the compiled optimizer. In the local code, the reconstruction loss consists of the summed squared error over the entire tensor plus the squared error of the feature-wise mean at each timestep. KL is summed over the latent dimension and batch; these losses are not normalized to a batch mean.

$$
\mathcal{L}=\texttt{reconstruction\_wt}\,\mathcal{L}_{rec}
+D_{KL}(q(z\mid x)\,\|\,\mathcal{N}(0,I)).
$$

`test_step` computes the same components without updating weights. `fit_on_data` calls Keras `fit` with the model's `batch_size`, EarlyStopping, and ReduceLROnPlateau; this API accepts only training data, `max_epochs`, and `verbose`.

## TimeVAE Decoder

| Component | Function/Class | Representation |
| --- | --- | --- |
| Level | `level_model` | Per-feature values repeated across the full sequence length |
| Trend | `TrendLayer` | Per-feature coefficients for powers of time from degree 1 to `trend_poly` |
| Seasonality | `SeasonalLayer` | Periodic patterns defined by `(num_seasons, len_per_season)` pairs |
| Residual | `_get_decoder_residual` | Neural decoder branch that models the remaining variation |

The decoder always includes a level component; the other three components are controlled through configuration. With the [default YAML](../config/hyperparameters.yaml), TimeVAE uses level and residual. No separate “generic seasonality” branch is constructed in `_get_decoder`, although some docstrings still mention the term.

## Reconstruction and Generation

| API | Behavior |
| --- | --- |
| `vae.predict(data)` / `get_posterior_samples(vae, data)` | `call` decodes `z_mean`; reconstruction uses the posterior mean rather than random posterior sampling |
| `get_prior_samples(vae, num_samples)` | Samples `Z ~ N(0,I)` and decodes it into new sequences |
| `vae.get_prior_samples_given_Z(Z)` | Decodes caller-provided latent values of shape `(N,latent_dim)` |
| `train_vae(vae, train_data, max_epochs, verbose=0)` | Calls `fit_on_data` |
| `save_vae_model(vae, dir_path)` / `load_vae_model(vae_type, dir_path)` | Saves/loads model construction parameters and encoder/decoder weights |

The sampling APIs return NumPy arrays in the data domain learned by the model; inverse scaling is handled by [data_utils.py](../data_utils.py). See the [usage example](../README.md) and [configuration](../config/README.md).

## Serialization and Usage Requirements

`save` writes `<model_name>_parameters.pkl` and two files, `*_encoder_wts.h5` and `*_decoder_wts.h5`. All are serialized using joblib, including files with the `.h5` extension. `load` reconstructs the architecture and compiles a new Adam optimizer; it does not restore the optimizer to resume the exact training state. `batch_size` is not included in the saved parameters.

For convolutional decoders, use at least two entries in `hidden_layer_sizes`: the code names the final convolution using the variable `i` from the preceding loop. The callbacks monitor `total_loss`, while `train_step` returns the key `loss`; check the actual metric name before relying on early stopping. See also the [integration notes](../../../docs/experimental-workflow.md).

## References

[TimeVAE: A Variational Auto-Encoder for Multivariate Time Series Generation](https://arxiv.org/abs/2111.08095) · [Authors' Repository](https://github.com/abudesai/timeVAE)
