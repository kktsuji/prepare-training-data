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


def _helper(input_path, output_dir, coeff):
    print("Processing...", input_path)
    output_path = output_dir + input_path.split("/")[-1]

    img = cv2.imread(input_path)
    img_gamma = _apply_gamma_correction(img, coeff)
    cv2.imwrite(output_path, img_gamma)


if __name__ == "__main__":
    dotenv.load_dotenv()
    BASE_NAME = os.getenv("BASE_NAME")
    FILE_NAME = os.getenv("FILE_NAME")
    SOURCE_DIR = f"{BASE_NAME}/{FILE_NAME}"
    OUT_DIR = f"{BASE_NAME}/{FILE_NAME}_inverse-gamma/"
    COEFF = 1 / 2.2  # inverse gamma correction
    JOBLIB_N_JOBS = -1

    input_path_list = glob.glob(f"{SOURCE_DIR}/*.png")
    os.makedirs(OUT_DIR, exist_ok=True)

    joblib.Parallel(n_jobs=JOBLIB_N_JOBS)(
        joblib.delayed(_helper)(input_path, OUT_DIR, COEFF)
        for input_path in input_path_list
    )
