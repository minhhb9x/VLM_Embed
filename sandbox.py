from src.arguments import ModelArguments, DataArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor, QWEN2_VL, VLM_IMAGE_TOKENS, Qwen2_VL_process_fn
from src.utils import batch_to_device
from PIL import Image
import torch
import os

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

model_args = ModelArguments(
    model_name='Qwen/Qwen2-VL-2B',
    checkpoint_path='TIGER-Lab/VLM2Vec-Qwen2VL-2B',
    pooling='last',
    normalize=True,
    model_backbone='qwen2_vl',
    lora=True
)
data_args = DataArguments()

processor = load_processor(model_args, data_args)
model = MMEBModel.load(model_args)
model = model.to('cuda', dtype=torch.float16)
model.eval()

def disable_lora_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.lower().find("lora") != -1:
            for attr in ["dropout", "lora_dropout"]:
                if hasattr(m, attr):
                    d = getattr(m, attr)
                    if hasattr(d, "p"):
                        d.p = 0.0
disable_lora_dropout(model)
# Batch processing
processor_inputs = {
    "text": [f'{VLM_IMAGE_TOKENS[QWEN2_VL]} Represent the given image with the following question: What is in the image',
          f'{VLM_IMAGE_TOKENS[QWEN2_VL]} Represent the given image with the following question: What is in the image'],
    "images": [Image.open('example.jpg'),
            Image.open('example.jpg')],
}
inputs = Qwen2_VL_process_fn(
    processor_inputs,
    processor, 
    square_padding=True)
inputs = batch_to_device(inputs, "cuda")
with torch.autocast(device_type="cuda", dtype=torch.float16):
    qry_output = model(qry=inputs)["qry_reps"]

processor_inputs = {
    "text": ['A cat and a dog', 'A cat and a tiger'],
    "images": [None, None],
}
inputs = Qwen2_VL_process_fn(
    processor_inputs,
    processor)
inputs = batch_to_device(inputs, "cuda")
with torch.autocast(device_type="cuda", dtype=torch.float16):
    tgt_output = model(tgt=inputs)["tgt_reps"]
print(model.compute_similarity(qry_output, tgt_output))
# tensor([[0.3316, 0.2900],
#         [0.3286, 0.2879]], device='cuda:0', grad_fn=<MmBackward0>)
