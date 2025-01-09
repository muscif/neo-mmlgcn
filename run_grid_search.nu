const base_conf = {
    n_layers: 0,
    device: cuda,
    batch_size: 2048,
    embedding_dim: 1024,
    learning_rate: 0.0005,
    dataset: dbbook,
    epochs: 150,
    top_k: 50,
    multimodal: true

    datasets:
        {
            dbbook: [images, text],
            ml1m: [audio, images, text, video]
        }
}

for n_layers in [0, 1, 2, 3, 4] {
    for embedding_dim in [64, 128, 256, 512, 1024] {
        for learning_rate in [0.001, 0.0005] {
            for top_k in [5, 10, 20, 50] {
                for multimodal in [false, true] {
                    mut conf = $base_conf
                    $conf = $conf |
                        update n_layers $n_layers |
                        update embedding_dim $embedding_dim |
                        update learning_rate $learning_rate |
                        update top_k $top_k |
                        update multimodal $multimodal

                    $conf | to toml | save -f config.toml
                    print $conf
                    py main.py
                }
            }
        }
    }
}
    