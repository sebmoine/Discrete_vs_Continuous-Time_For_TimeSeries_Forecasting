from .models import *

def build_model(cfg, input_size, num_classes):
    modelname = cfg['class']
    return eval(f"{modelname}(cfg, input_size, num_classes)")