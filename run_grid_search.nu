let base_conf = open config.toml

for n_layers in [0, 1, 2, 3] {
    for embedding_dim in [64, 128, 256, 512, 1024, 2048] {
        for fusion in ["max", "mean"] {
            for multimodal in [false, true] {
                mut conf = $base_conf
                $conf = $conf |
                    update n_layers $n_layers |
                    update embedding_dim $embedding_dim |
                    update multimodal $multimodal
    
                $conf | to toml | save -f config.toml
                py main.py
            }
        }
    }
}
    