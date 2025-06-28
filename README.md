# Multimodal LightGCN
This project uses the [uv](https://docs.astral.sh/uv/) package and project manager.

To run, run ```src/main.py```; configuration will be loaded from ```src/config.toml```; the GPU with the lowest used memory will be used.

```logs/``` contains the logs of the training with the metrics for each epoch; the first line contains a JSON representation of the configuration used to produce that file.