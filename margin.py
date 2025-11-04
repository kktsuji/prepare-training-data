import glob
import os

import cv2
import numpy as np


def _margin(img, margin):

    h, w = img.shape[:2]
    dst = cv2.resize(img, (w // 8, h // 8))
    # draw lines rectangle, centered, width = margin * 2, height = margin * 2
    # cv2.rectangle(dst, (margin, margin), (w - margin, h - margin), (0, 255, 0), 2)
    center_h, center_w = h // 16, w // 16
    dst[
        center_h - margin : center_h + margin, center_w - margin : center_w + margin
    ] = 0

    return dst


if __name__ == "__main__":
    _MARGIN = 16

    method = "bilinear"
    if method == "bilinear":
        interpolation = cv2.INTER_LINEAR
    elif method == "nearest":
        interpolation = cv2.INTER_NEAREST
    elif method == "bicubic":
        interpolation = cv2.INTER_CUBIC
    else:
        raise ValueError("Unknown interpolation method")

    input_path_list = glob.glob(f"./in/in-{method}-512/*")
    out_dir = f"./out/out-{method}-512-margin/"
    os.makedirs(out_dir, exist_ok=True)

    for input_path in input_path_list:
        print("Processing...", input_path)
        output_path = out_dir + input_path.split("/")[-1]

        img = cv2.imread(input_path)
        img_adjusted = _margin(img, _MARGIN)
        cv2.imwrite(output_path, img_adjusted)
        cv2.imwrite(output_path, img_adjusted)
        cv2.imwrite(output_path, img_adjusted)
