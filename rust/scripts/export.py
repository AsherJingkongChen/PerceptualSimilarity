"""Write the weights LPIPS alex runs on as one safetensors file, and print the distances the Rust port must match.

    uv run python export.py OUT.safetensors original.png modified.png [modified.png ...]

The trunk is torchvision's AlexNet IMAGENET1K_V1 features, the head is lpips/weights/v0.1/alex.pth. safetensors is
written by hand (an 8-byte header length, a JSON header, the raw little-endian f32s) so nothing new is installed.
"""
import json
import struct
import sys
import time

import torch

import lpips

out, original, *modified = sys.argv[1:]
metric = lpips.LPIPS(net="alex", verbose=False).eval()
tensors = {k: v.detach().to(torch.float32).contiguous() for k, v in metric.state_dict().items()}
header, blobs, offset = {}, [], 0
for name, t in sorted(tensors.items()):
    blob = t.numpy().astype("<f4").tobytes()
    header[name] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [offset, offset + len(blob)]}
    blobs.append(blob)
    offset += len(blob)
head = json.dumps(header, separators=(",", ":")).encode()
head += b" " * (-len(head) % 8)
with open(out, "wb") as f:
    f.write(struct.pack("<Q", len(head)) + head + b"".join(blobs))
print(f"{len(tensors)} tensors, {offset / 1e6:.1f} MB", file=sys.stderr)

torch.set_num_threads(torch.get_num_threads())
with torch.inference_mode():
    x = lpips.im2tensor(lpips.load_image(original))
    for path in modified:
        y = lpips.im2tensor(lpips.load_image(path))
        metric(x, y)
        t0 = time.perf_counter()
        d = metric(x, y).item()
        ms = (time.perf_counter() - t0) * 1e3
        print(f"{d:.10f}\t{path}\t{ms:.0f} ms")
