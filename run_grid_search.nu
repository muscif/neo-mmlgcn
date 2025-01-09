let base_conf = open config.toml

const l_n_layers = [0, 4]
const l_embedding_dim = [1024, 2048, 4096]
const l_fusion = ["max", "mean"]
const l_multimodal = [false, true]
const l_single_branch = [false, true]

mut i = 0
let tot = (($l_n_layers | length) * ($l_embedding_dim | length) * ($l_fusion | length) * ($l_multimodal | length)) | into string

for n_layers in $l_n_layers {
    for embedding_dim in $l_embedding_dim {
        for fusion in $l_fusion {
            for multimodal in $l_multimodal {
                print ("PROGRESS: " + ($i|into string) + "/" + $tot)

                mut conf = $base_conf
                $conf = $conf |
                    update n_layers $n_layers |
                    update embedding_dim $embedding_dim |
                    update fusion $fusion |
                    update multimodal $multimodal
    
                $conf | to toml | save -f config.toml
                py main.py
                $i += 1
            }
        }
    }
}
    