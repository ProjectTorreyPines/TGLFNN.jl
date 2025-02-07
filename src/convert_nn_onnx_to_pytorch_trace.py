#!/usr/bin/env python
# coding: utf-8

# Takes all .onnx files in working directory and converts them to
# .pt PyTorch traces. Assumes 'xm.txt' is present and can be used
# as example model input (to generate trace).

import onnx
import torch
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
example_input.shape

# Loop over model files; convert to pytorch,
# run with example input to generate trace, then save to trace .pt file.
for model_file_in in onnx_model_files:
    print(model_file_in)
    onnx_model    = onnx.load(model_file_in)
    pytorch_model = ConvertModel(onnx_model)
    traced_model  = torch.jit.trace(pytorch_model,example_input)
    ofile = os.path.splitext(model_file_in)[0]+".pt"
    print(ofile)
    traced_model.save(ofile)
    print(traced_model.forward(example_input)) # check for valid output
