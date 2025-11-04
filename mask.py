import glob
import os

import cv2
import joblib
import numpy as np

_MARGIN = 16
_METHODS = [
    "rectangle",
    "circle",
    "circle-distance",
    "rectangle-inverse",
    "circle-inverse",
    "circle-distance-inverse",
]
_FILENAMES = [
    "xxx",
]


def _margin(img, method):

    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    margin_size = _MARGIN * 8
    center_h, center_w = h // 2, w // 2

    if method == _METHODS[0] or method == _METHODS[3]:  # rectangle or rectangle-inverse
        mask[
            center_h - margin_size : center_h + margin_size,
            center_w - margin_size : center_w + margin_size,
        ] = 1  # foreground: 1, background: 0

        if method == _METHODS[3]:  # rectangle-inverse
            mask = 1 - mask  # inverse

    elif method == _METHODS[1] or method == _METHODS[4]:  # circle or circle-inverse
        for y, x in np.ndindex(mask.shape[:2]):
            dist_y = abs(y - center_h)
            dist_x = abs(x - center_w)
            dist = np.sqrt(dist_y**2 + dist_x**2)
            if dist < margin_size:
                mask[y, x] = 1
            else:
                mask[y, x] = 0

        if method == _METHODS[4]:  # circle-inverse
            mask = 1 - mask  # inverse

    elif (
        method == _METHODS[2] or method == _METHODS[5]
    ):  # circle-distance or circle-distance-inverse
        mask = np.zeros_like(img, dtype=np.float32)
        dist_max = center_h  # np.sqrt(center_h**2 + center_w**2)
        for y, x in np.ndindex(mask.shape[:2]):
            dist_y = abs(y - center_h)
            dist_x = abs(x - center_w)
            dist = np.sqrt(dist_y**2 + dist_x**2)
            if dist < margin_size:
                mask[y, x] = 1.0
            else:
                mask[y, x] = 1.0 - (dist - margin_size) / (dist_max - margin_size)

        if method == _METHODS[5]:  # circle-distance-inverse
            mask = 1 - mask  # inverse

    return img * mask


def helper(input_path, out_dir, method):
    print("Processing...", input_path)
    output_path = out_dir + input_path.split("/")[-1]

    img = cv2.imread(input_path)
    img_adjusted = _margin(img, method)
    cv2.imwrite(output_path, img_adjusted)


if __name__ == "__main__":

    for base_name in _FILENAMES:
        for method in _METHODS:
            input_path_list = glob.glob(f"./out/{base_name}/*")
            out_dir = f"./out/{base_name}_background-mask-{method}/"
            os.makedirs(out_dir, exist_ok=True)

            joblib.Parallel(n_jobs=10)(
                joblib.delayed(helper)(input_path, out_dir, method)
                for input_path in input_path_list
            )

            # for input_path in input_path_list:
            #     print("Processing...", input_path)
            #     output_path = out_dir + input_path.split("/")[-1]

            #     img = cv2.imread(input_path)
            #     img_adjusted = _margin(img, _MARGIN)
            #     cv2.imwrite(output_path, img_adjusted)
