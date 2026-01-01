import numpy as np
import torch

from src.model.model import MMEBModel
from src.model.processor import LLAVA_NEXT, QWEN2_VL, PHI3V, print_master, QWEN2_5_VL, \
    QWEN2_VL_TOKENSELECTION, backbone2model, GME, VLM_IMAGE_TOKENS, LamRA, \
    COLPALI, INTERN_VL3, LLAVA_ONEVISION, LLAVA_QWEN2


def get_grid_size(model: MMEBModel, inputs):
    if model.model_backbone == LLAVA_QWEN2:
        vision_tower = model.encoder.get_vision_tower()
        vision_config = vision_tower.config
        patch_size = vision_config['image_cfg']['patch_size']
        grid_sizes = []
        for image in inputs['images']:
            h, w = image.shape[-2:]
            grid_h = h // patch_size
            grid_w = w // patch_size
            grid_sizes.append((grid_h, grid_w))
        return grid_sizes
    elif model.model_backbone in [QWEN2_VL, QWEN2_5_VL]:
        vision_config = model.config.vision_config
        merge_size = vision_config.spatial_merge_size
        grid_sizes = []
        for shape in inputs['image_grid_thw']:
            h, w = shape[0, -2:]
            grid_h = (h // merge_size).item()
            grid_w = (w // merge_size).item()
            grid_sizes.append((grid_h, grid_w))
        return grid_sizes

