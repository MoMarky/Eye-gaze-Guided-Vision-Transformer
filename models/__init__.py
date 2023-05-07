from .EGViT import *

def build_model_v1(layer, num_class):
    return MaskViT(layer=layer, num_classes=num_class)
