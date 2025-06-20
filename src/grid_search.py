from utils import CONFIG
from main import main
import GPUtil

if not CONFIG.log:
    raise Exception("Enable logging before continuing!")

gpu_id = GPUtil.getAvailable(order="memory", limit=10, maxLoad=1, maxMemory=1)[0]

CONFIG.multimodal = False
for n_layers in  [3, 2, 1, 0]:
    CONFIG.n_layers = n_layers

    main(gpu_id)

CONFIG.multimodal = True
for n_layers in [3, 2, 1, 0]:
    CONFIG.n_layers = n_layers
    CONFIG.weighting = False

    for freeze in [False, True]:
        CONFIG.freeze = freeze
        main(gpu_id)

    CONFIG.freeze = freeze
    for weighting in ["alpha", "normalized", "equal"]:
        CONFIG.weighting = weighting
        main(gpu_id)
    