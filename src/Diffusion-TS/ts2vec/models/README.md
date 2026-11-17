# TS2Vec Encoder and Contrastive Loss

PyTorch building blocks for [TS2Vec](../README.md): transforming sequences of shape `(B,L,F)` into representations of shape `(B,L,D)` and learning consistency between two views of the same data.

## Components

| File | API | Responsibility |
| --- | --- | --- |
| [encoder.py](encoder.py) | `TSEncoder(input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial')` | Linear projection, timestamp masking, convolution, and dropout |
| [dilated_conv.py](dilated_conv.py) | `SamePadConv`, `ConvBlock`, `DilatedConvEncoder` | Length-preserving convolution and residual blocks with dilation increasing as `2**i` |
| [losses.py](losses.py) | `hierarchical_contrastive_loss`, `instance_contrastive_loss`, `temporal_contrastive_loss` | Contrast across instances and timestamps, aggregated across multiple resolutions |

`TSEncoder.forward(x, mask=None)` projects features to the hidden dimension, applies a mask, changes the layout to `(B,H,L)` for Conv1D, and then restores it to `(B,L,D)`. `depth` creates the list `[hidden_dims] * depth + [output_dims]`, which includes an additional output block; do not interpret `depth=10` as exactly ten Conv1D layers. Each `ConvBlock` contains two convolutions and a residual branch.

## Masking

The default mask is `binomial` during training and `all_true` during evaluation. Other options are `continuous`, `all_false`, and `mask_last`; `True` keeps a timestamp, while `False` masks it. `generate_binomial_mask` uses a default keep probability of `0.5`, while continuous masking hides consecutive segments.

A NaN in any feature excludes the entire timestamp from the effective input. The encoder sets these positions to 0 directly in the input tensor; callers should account for this when reusing the tensor. Masking is applied before convolution and does not guarantee that the output vector at a masked position will be zero.

## Objective Function

`hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0)` takes two representations of the overlapping region with the same shape `(B,L,D)`. At each level, it combines:

- **Instance contrast:** compares two views of the same instance against other instances at the same position; this component is 0 when `B=1`.
- **Temporal contrast:** distinguishes temporal positions; starts at level `temporal_unit` and is 0 when only one timestamp remains.

After each level, max pooling reduces the temporal dimension by a factor of 2; the loss is divided by the number of levels considered. `alpha` controls the balance between the two components; the result is a scalar used for backpropagation.

The module does not include a training loop, whole-sequence pooling, or model saving. These responsibilities belong to [TS2Vec](../ts2vec.py). Relative imports in `encoder.py` require importing it as a module/package rather than running the file directly as a script.

## Further Reading

[TS2Vec: Method and API](../README.md) · [Paper](https://arxiv.org/abs/2106.10466) · [Authors' Repository](https://github.com/zhihanyue/ts2vec)
