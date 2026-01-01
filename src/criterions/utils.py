import numpy as np
import torch

from src.model.model import MMEBModel
from src.model.processor import LLAVA_NEXT, QWEN2_VL, PHI3V, print_master, QWEN2_5_VL, \
    QWEN2_VL_TOKENSELECTION, backbone2model, GME, VLM_IMAGE_TOKENS, LamRA, \
    COLPALI, INTERN_VL3, LLAVA_ONEVISION, LLAVA_QWEN2


def get_hidden_text_vision(hidden_state, num_text_token, num_vision_token, model_backbone):
    '''
    Get hidden states for text and vision tokens separately
    Args:
        hidden_state: tensor, the output hidden states from the model
        num_text_token: int, number of text tokens
        num_vision_token: int, number of vision tokens
        model_backbone: str, the model backbone type
        (note: only )
    '''
    if model_backbone in [QWEN2_VL, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION]: # left padding
        vision_hidden_state = hidden_state[-(num_vision_token+num_text_token): -num_text_token, :]
        text_hidden_state = hidden_state[-num_text_token:, :]
    elif model_backbone == LLAVA_QWEN2: # right padding
        vision_hidden_state = hidden_state[:num_vision_token, :]
        text_hidden_state = hidden_state[num_vision_token: num_vision_token + num_text_token, :]
    else:
        raise NotImplementedError(f"get_hidden_text_vision not implemented for model_backbone {model_backbone}")
    return text_hidden_state, vision_hidden_state

def get_grid_size(model: MMEBModel, inputs):
    if model.model_backbone == LLAVA_QWEN2:
        vision_tower = model.encoder.get_vision_tower()
        vision_config = vision_tower.config
        patch_size = vision_config['image_cfg']['patch_size']
        grid_sizes = []
        if 'images' not in inputs:
            return grid_sizes
        for image in inputs['images']:
            if image is None:
                continue
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
            if shape is None:
                continue
            h, w = shape[0, -2:]
            grid_h = (h // merge_size).item()
            grid_w = (w // merge_size).item()
            grid_sizes.append((grid_h, grid_w))
        return grid_sizes

