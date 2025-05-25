from utils import CONFIG
from main import main
import itertools

if not CONFIG.log:
    raise Exception("Enable logging before continuing!")

l_n_layers = [0,1,2,3]
#l_single_branch = [False, True]
l_freeze = [False, True]
#l_autoencoder = [False, True]

if False:
    CONFIG.multimodal = False
    for comb in itertools.product(l_n_layers):
        n_layers = comb[0]

        CONFIG.n_layers = n_layers

        main()

if True:
    CONFIG.multimodal = True
    for comb in itertools.product(l_n_layers):
        n_layers = comb[0]

        CONFIG.n_layers = n_layers

        main()
