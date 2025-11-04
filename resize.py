import glob
import os

import cv2
import dotenv
import joblib
import numpy as np


def _apply_gamma_correction(img, gamma=2.2):
    img_normalized = img.astype(np.float32) / 255.0
    img_corrected = np.power(img_normalized, 1 / gamma)
    return (img_corrected * 255).astype(np.uint8)


def _helper(input_path, output_dir, width, height, interpolation):
    print("Processing...", input_path)
    output_path = output_dir + input_path.split("/")[-1]

    img = cv2.imread(input_path)
    img_resized = cv2.resize(img, (width, height), interpolation=interpolation)
    cv2.imwrite(output_path, img_resized)


if __name__ == "__main__":
    dotenv.load_dotenv()
    BASE_NAME = os.getenv("BASE_NAME")
    FILE_NAME = os.getenv("FILE_NAME")
    SOURCE_DIR = f"{BASE_NAME}/{FILE_NAME}"
    HEIGHT = 40
    WIDTH = 40
    OUT_DIR = f"{BASE_NAME}/{FILE_NAME}_resized-{WIDTH}x{HEIGHT}/"
    JOBLIB_N_JOBS = -1

    method = "bilinear"
    if method == "bilinear":
        interpolation = cv2.INTER_LINEAR
    elif method == "nearest":
        interpolation = cv2.INTER_NEAREST
    elif method == "bicubic":
        interpolation = cv2.INTER_CUBIC
    else:
        raise ValueError("Unknown interpolation method")

    input_path_list = glob.glob(f"{SOURCE_DIR}/*.png")
    os.makedirs(OUT_DIR, exist_ok=True)

    joblib.Parallel(n_jobs=JOBLIB_N_JOBS)(
        joblib.delayed(_helper)(input_path, OUT_DIR, WIDTH, HEIGHT, interpolation)
        for input_path in input_path_list
    )
