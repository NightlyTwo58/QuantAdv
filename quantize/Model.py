import torch
from torchao.quantization import quantize_, Int8WeightOnlyConfig
import copy

class Model:
    def __init__(self, model):
        # base model
        self.model = model

        # int8 PTQ is a simple quantization of the base FP32 model.
        self.int8_PTQ = quantize_(
            model=copy.deepcopy(model),
            config=Int8WeightOnlyConfig(),
            device="cuda" if torch.cuda.is_available() else "cpu"
            )
        
        self.int8_QAT = None

    def prepare_QAT():