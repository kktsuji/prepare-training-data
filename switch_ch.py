import glob
import os

import cv2
import numpy as np

src_dir = "./data/in/"
dst_dir_bgr = "./out_switched-ch/bgr/"
dst_dir_grb = "./out_switched-ch/grb/"
dst_dir_gray_r = "./out_switched-ch/gray_r/"

os.makedirs(dst_dir_bgr, exist_ok=True)
os.makedirs(dst_dir_grb, exist_ok=True)
os.makedirs(dst_dir_gray_r, exist_ok=True)

for fp in glob.glob(f"{src_dir}/*.png"):
    print("Processing...", fp)
    basename = os.path.basename(fp)
    img = cv2.imread(fp)

    # bgr to rgb
    img_switched = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{dst_dir_bgr}/{basename}", img_switched)

    # bgr to grb
    img_switched = img[:, :, [1, 2, 0]]
    cv2.imwrite(f"{dst_dir_grb}/{basename}", img_switched)

    # gray_r
    img_gray_r = np.stack([img[:, :, 2]] * 3, axis=-1)
    cv2.imwrite(f"{dst_dir_gray_r}/{basename}", img_gray_r)
