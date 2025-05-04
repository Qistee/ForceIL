from PIL import Image
import torch
from zoedepth.utils.misc import colorize

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

# repo = "isl-org/ZoeDepth"
# # Zoe_N
# model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)


# # Zoe_N
# model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)
# ZoeD_N
conf = get_config("zoedepth", "infer")
model_zoe_n = build_model(conf)


image = Image.open("middle2.jpg").convert("RGB")  # load
image=image.resize([384,384])
# depth_numpy = zoe.infer_pil(image)  # as numpy

# depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image

depth_tensor = model_zoe_n.infer_pil(image)  # as torch tensor

print(depth_tensor.size)
colored = colorize(depth_tensor)
print(colored.size)