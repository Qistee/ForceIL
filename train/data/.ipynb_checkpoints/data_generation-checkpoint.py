import h5py
import numpy as np
from tqdm import trange, tqdm
from PIL import Image
import torch

# ─── 1. 加载 ZoeDepth ────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)  \
               .to(DEVICE).eval()           # 单张推理接口：zoe.infer_pil(img)

def infer_depth(rgb_np: np.ndarray) -> np.ndarray:
    """
    rgb_np : uint8 H×W×3 (RGB 顺序)
    return : float32 H×W  (以米为单位的 metric depth)
    """
    img = Image.fromarray(rgb_np)
    depth = zoe.infer_pil(img)              # numpy, float32
    return depth

# ─── 2. 打开 HDF5、遍历图像、写回深度 ──────────────────────────────────────────────
folder="/root/autodl-tmp/data"
folder = Path(folder).expanduser().resolve()
# h5_path = ".h5"
for h5_path in folder.rglob("*"):
    if h5_path.suffix.lower() in suffixes:
            
        with h5py.File(h5_path, "r+") as f:
            img_grp = f["image_logitech"]

            # 若已有旧结果可先删除；也可改成 require_group 追加
            if "depth_logitech" in f:
                del f["depth_logitech"]
            depth_grp = f.create_group("depth_logitech")

            # 2‑A) image_logitech 直接是 N×H×W×3 的 dataset -----------------------------
            if isinstance(img_grp, h5py.Dataset):
                images = img_grp                      # 不立即[:]，节省内存
                N = images.shape[0]
                for i in trange(N, desc="ZoeDepth"):
                    depth = infer_depth(images[i])
                    depth_grp.create_dataset(
                        str(i), data=depth, dtype="float32",
                        compression="gzip", compression_opts=4
                    )

            # 2‑B) image_logitech 下面是一个个子‑dataset ---------------------------------
            else:
                for name in tqdm(img_grp.keys(), desc="ZoeDepth"):
                    rgb_np = img_grp[name][:]
                    depth  = infer_depth(rgb_np)
                    depth_grp.create_dataset(
                        name, data=depth, dtype="float32",
                        compression="gzip", compression_opts=4
                    )

        print("Done ✓ 所有深度图已写回 depth_logitech 组")

        with h5py.File(h5_path, "r+") as f:
            img_grp = f["image_realsense"]

            # 若已有旧结果可先删除；也可改成 require_group 追加
            if "depth_realsense" in f:
                del f["depth_realsense"]
            depth_grp = f.create_group("depth_realsense")

            # 2‑A) image_logitech 直接是 N×H×W×3 的 dataset -----------------------------
            if isinstance(img_grp, h5py.Dataset):
                images = img_grp                      # 不立即[:]，节省内存
                N = images.shape[0]
                for i in trange(N, desc="ZoeDepth"):
                    depth = infer_depth(images[i])
                    depth_grp.create_dataset(
                        str(i), data=depth, dtype="float32",
                        compression="gzip", compression_opts=4
                    )

            # 2‑B) image_logitech 下面是一个个子‑dataset ---------------------------------
            else:
                for name in tqdm(img_grp.keys(), desc="ZoeDepth"):
                    rgb_np = img_grp[name][:]
                    depth  = infer_depth(rgb_np)
                    depth_grp.create_dataset(
                        name, data=depth, dtype="float32",
                        compression="gzip", compression_opts=4
                    )

        print("Done ✓ 所有深度图已写回 depth_realsense 组")
