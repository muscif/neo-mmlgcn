from utils import CONFIG
from main import main
import itertools

if not CONFIG.log:
    raise Exception("Enable logging before continuing!")

l_n_layers = [0,1,2,3,4]
l_embedding_dim = [64, 128, 256]
#l_fusion = ["mean", "max"]
#l_single_branch = [False, True]
#l_freeze = [False, True]
#l_autoencoder = [False, True]
l_bidirectional = [False, True]

CONFIG.multimodal = False
for comb in itertools.product(l_n_layers, l_embedding_dim, l_bidirectional):
    n_layers, embedding_dim, bidirectional = comb

    CONFIG.n_layers = n_layers
    CONFIG.embedding_dim = embedding_dim
    CONFIG.bidirectional = bidirectional

    main()

CONFIG.multimodal = True
for comb in itertools.product(l_n_layers, l_embedding_dim, l_bidirectional):
    n_layers, embedding_dim, bidirectional = comb

    CONFIG.n_layers = n_layers
    CONFIG.embedding_dim = embedding_dim
    CONFIG.bidirectional = bidirectional

    main()
