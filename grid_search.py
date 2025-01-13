from utils import CONFIG
from main import main
import itertools

if not CONFIG.log:
    raise Exception("Enable logging before continuing!")

l_n_layers = [0,1,2,3,4]
l_embedding_dim = [512, 1024, 2048]
l_fusion = ["concat", "mean", "sum", "max", "min", "prod"]
l_single_branch = [False, True]
l_freeze = [False, True]
l_autoencoder = [False, True]

CONFIG.multimodal = False
for comb in itertools.product(l_n_layers, l_embedding_dim):
    n_layers, embedding_dim = comb
    CONFIG.n_layers = n_layers
    CONFIG.embedding_dim = embedding_dim

    main()

CONFIG.multimodal = True
for comb in itertools.product(l_n_layers, l_embedding_dim, l_fusion, l_autoencoder, l_freeze):
    n_layers, embedding_dim, fusion, autoencoder, freeze, multimodal = comb

    CONFIG.n_layers = n_layers
    CONFIG.embedding_dim = embedding_dim
    CONFIG.fusion_modalities = fusion
    CONFIG.autoencoder = autoencoder
    CONFIG.freeze = freeze
    CONFIG.multimodal = multimodal

    main()
