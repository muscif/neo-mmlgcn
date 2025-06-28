# Multimodal LightGCN
This project enhances [LightGCN](https://arxiv.org/abs/2002.02126) with multimodal embeddings.

Various architectures of fusion of collaborative and multimodal embeddings are tested.

The results show that multimodal information is most effective in settings where collaborative information is scarce.

# Running
This project uses the [uv](https://docs.astral.sh/uv/) package and project manager.

```src/config.toml``` contains the configuration for the model. Attributes and possible values:
- `fusion_modalities`: `concat`|`mean`|`sum`|`max`|`min`|`prod`
- `fusion_type`: `late`|`early`|`inner`
- `weighting`: `false`|`alpha`|`normalized`|`equal`

To run, run ```src/main.py```; configuration will be loaded from ```src/config.toml```; the GPU with the lowest used memory will be used.

```logs/``` contains the logs of the training with the metrics for each epoch; the first line contains a JSON representation of the configuration used to produce that file.

`src/graph.ipynb` contains utilities to read and filter the logs, show them in tabular format and produce graphs.