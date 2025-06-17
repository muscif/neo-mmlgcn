from utils import CONFIG
from main import main
import itertools

if not CONFIG.log:
    raise Exception("Enable logging before continuing!")

l_n_layers = [0,1,2,3]
l_freeze = [False, True]
l_weighting = [False, "alpha", "normalized", "equal"]

if True:
    CONFIG.multimodal = False
    for layer in l_n_layers:
        CONFIG.n_layers = layer

        main()

if True:
    CONFIG.multimodal = True
    for n_layers, freeze, weighting in itertools.product(l_n_layers, l_freeze, l_weighting):
        CONFIG.n_layers = n_layers
        CONFIG.freeze = freeze
        CONFIG.weighting = weighting

        main()
