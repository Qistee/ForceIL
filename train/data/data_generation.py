# import h5py
# import numpy as np
# from tqdm import trange, tqdm
# from PIL import Image
# import torch

# # ─── 1. 加载 ZoeDepth ────────────────────────────────────────────────────────────
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)  \
#                .to(DEVICE).eval()           # 单张推理接口：zoe.infer_pil(img)

# def infer_depth(rgb_np: np.ndarray) -> np.ndarray:
#     """
#     rgb_np : uint8 H×W×3 (RGB 顺序)
#     return : float32 H×W  (以米为单位的 metric depth)
#     """
#     img = Image.fromarray(rgb_np)
#     depth = zoe.infer_pil(img)              # numpy, float32
#     return depth

# # ─── 2. 打开 HDF5、遍历图像、写回深度 ──────────────────────────────────────────────
# folder="/root/autodl-tmp/data"
# folder = Path(folder).expanduser().resolve()
# # h5_path = ".h5"
# for h5_path in folder.rglob("*"):
#     if h5_path.suffix.lower() in suffixes:
            
#         with h5py.File(h5_path, "r+") as f:
#             img_grp = f["image_logitech"]

#             # 若已有旧结果可先删除；也可改成 require_group 追加
#             if "depth_logitech" in f:
#                 del f["depth_logitech"]
#             depth_grp = f.create_group("depth_logitech")

#             # 2‑A) image_logitech 直接是 N×H×W×3 的 dataset -----------------------------
#             if isinstance(img_grp, h5py.Dataset):
#                 images = img_grp                      # 不立即[:]，节省内存
#                 N = images.shape[0]
#                 for i in trange(N, desc="ZoeDepth"):
#                     depth = infer_depth(images[i])
#                     depth_grp.create_dataset(
#                         str(i), data=depth, dtype="float32",
#                         compression="gzip", compression_opts=4
#                     )

#             # 2‑B) image_logitech 下面是一个个子‑dataset ---------------------------------
#             else:
#                 for name in tqdm(img_grp.keys(), desc="ZoeDepth"):
#                     rgb_np = img_grp[name][:]
#                     depth  = infer_depth(rgb_np)
#                     depth_grp.create_dataset(
#                         name, data=depth, dtype="float32",
#                         compression="gzip", compression_opts=4
#                     )

#         print("Done ✓ 所有深度图已写回 depth_logitech 组")

#         with h5py.File(h5_path, "r+") as f:
#             img_grp = f["image_realsense"]

#             # 若已有旧结果可先删除；也可改成 require_group 追加
#             if "depth_realsense" in f:
#                 del f["depth_realsense"]
#             depth_grp = f.create_group("depth_realsense")

#             # 2‑A) image_logitech 直接是 N×H×W×3 的 dataset -----------------------------
#             if isinstance(img_grp, h5py.Dataset):
#                 images = img_grp                      # 不立即[:]，节省内存
#                 N = images.shape[0]
#                 for i in trange(N, desc="ZoeDepth"):
#                     depth = infer_depth(images[i])
#                     depth_grp.create_dataset(
#                         str(i), data=depth, dtype="float32",
#                         compression="gzip", compression_opts=4
#                     )

#             # 2‑B) image_logitech 下面是一个个子‑dataset ---------------------------------
#             else:
#                 for name in tqdm(img_grp.keys(), desc="ZoeDepth"):
#                     rgb_np = img_grp[name][:]
#                     depth  = infer_depth(rgb_np)
#                     depth_grp.create_dataset(
#                         name, data=depth, dtype="float32",
#                         compression="gzip", compression_opts=4
#                     )

#         print("Done ✓ 所有深度图已写回 depth_realsense 组")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_actions.py
在所有 HDF5 文件中新建 /action (N,7) = [joint_pos, gripper_pos]
"""

from pathlib import Path
import numpy as np
import h5py

# ────────── 配 置 ──────────
ROOT_DIR = Path("/root/autodl-tmp/grasp_with_force_feeedback")   # 根目录（递归搜索）
JOINT_KEY   = "/actions/joint_pos"           # (N,6)
GRIPPER_KEY = "/actions/gripper_pos"         # (N,1)
OUT_KEY     = "/action"                      # (N,7)

def merge_single_file(h5_path: Path):
    """在单个 HDF5 文件里生成 /action 数据集"""
    print(f"处理 {h5_path.relative_to(ROOT_DIR)}")

    with h5py.File(h5_path, "r+") as f:

        # 1) 读取原数据
        try:
            joint  = f[JOINT_KEY][:]
            grip   = f[GRIPPER_KEY][:]
        except KeyError as e:
            print(f"  ⚠ 缺少数据集 {e}; 跳过")
            return

        if joint.shape[0] != grip.shape[0]:
            print("  ⚠ 尺寸不匹配，跳过")
            return

        # 2) 拼接 (N,7)
        action = np.concatenate([joint, grip], axis=1)  # (N,6+1)

        # 3) 若已存在旧 /action，先删除
        if OUT_KEY in f:
            del f[OUT_KEY]

        # 4) 写入，可选压缩
        f.create_dataset(
            OUT_KEY,
            data=action.astype(np.float32),
            compression="gzip",
            compression_opts=4
        )
        print("  ✓ 已写入 /action")

def main():
    # 遍历所有 .h5 / .hdf5
    print("begin")
    for h5_path in ROOT_DIR.rglob("*"):
        print(h5_path)
        if h5_path.suffix.lower() in {".h5", ".hdf5"}:
            merge_single_file(h5_path)

if __name__ == "__main__":
    main()
