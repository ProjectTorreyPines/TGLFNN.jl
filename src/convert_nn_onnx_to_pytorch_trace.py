#!/usr/bin/env python
# coding: utf-8

# Takes all .onnx files in working directory and converts them to a single
# .pt file of comittee PyTorch trace. Assumes 'xm.txt' is present and can
# be used as example model input (to generate trace).

import onnx
import torch
import torch.nn as nn
import numpy as np
from onnx2pytorch import ConvertModel
import os

# Get list of onnx model files.
path = os.getcwd()
onnx_model_files = [f for f in os.listdir(path) if f.endswith('.onnx')]

# Load example input data used to trace models.
example_as_np = np.loadtxt('xm.txt')
N_inputs = example_as_np.size
example_input = torch.as_tensor(example_as_np.reshape(1,N_inputs),dtype=torch.float32) # Construct as (1,N_inputs) tensor.

pytorch_models = []  # List to store the converted PyTorch models

for model_file_in in onnx_model_files:
    print(model_file_in)
    onnx_model = onnx.load(model_file_in)
    pytorch_model = ConvertModel(onnx_model)
    pytorch_models.append(pytorch_model) # Add to the list

# Create the Committee module
class Committee(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models) # Important: Use ModuleList

    def gather_models(self, input):
        outputs = []
        for model in self.models:
            outputs.append(model(input))
        return torch.stack(outputs)  # Returns a tensor of shape (num_models, *output_shape)

    def forward(self, input):
        all_outputs = self.gather_models(input)
        return torch.mean(all_outputs, dim=0), torch.var(all_outputs, dim=0)

committee = Committee(pytorch_models)

# Trace the committee
traced_committee = torch.jit.trace(committee, example_input)

# Save the TorchScript committee
output_filename = "committee.pt"
traced_committee.save(output_filename)
print(f"Committee TorchScript saved to {output_filename}")
