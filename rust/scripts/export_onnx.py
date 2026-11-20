"""Write LPIPS alex as an ONNX graph for burn-onnx: two 1 x 3 x SIZE x SIZE images in [-1, 1] in, one distance out.

    uv run python export_onnx.py OUT.onnx SIZE
"""
import sys

import torch

import lpips

out, size = sys.argv[1], int(sys.argv[2])
metric = lpips.LPIPS(net="alex", verbose=False).eval()
x = torch.zeros(1, 3, size, size)
torch.onnx.export(metric, (x, x), out, dynamo=False, opset_version=17,
                  input_names=["original", "modified"], output_names=["distance"])
print(out)
