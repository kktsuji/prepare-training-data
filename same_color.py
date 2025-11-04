import glob
import os

import cv2
import numpy as np

COLORS = ["b", "g", "r"]

if __name__ == "__main__":

    input_name = "xxx"
    input_path_list = glob.glob(f"./out/{input_name}/*")
    out_dir = f"./out/{input_name}_same-color-{{}}/"
    for color in COLORS:
        os.makedirs(out_dir.format(color), exist_ok=True)

    for input_path in input_path_list:
        print("Processing...", input_path)

        img = cv2.imread(input_path)

        for i, color in enumerate(COLORS):
            output_path = out_dir.format(color) + input_path.split("/")[-1]
            dst = np.zeros_like(img)
            dst[:, :, 0] = img[:, :, i]
            dst[:, :, 1] = img[:, :, i]
            dst[:, :, 2] = img[:, :, i]

            cv2.imwrite(output_path, dst)
            cv2.imwrite(output_path, dst)
            cv2.imwrite(output_path, dst)
