import glob
import os

import cv2
import dotenv
import joblib
import numpy as np


def _contrast_stretch(img, lower_percentile=1, upper_percentile=99):
    # convert rgb to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # apply contrast stretching on the V channel
    lower = np.percentile(v, lower_percentile)
    upper = np.percentile(v, upper_percentile)
    v_stretched = np.clip((v - lower) * (255.0 / (upper - lower)), 0, 255)
    v_stretched = v_stretched.astype(np.uint8)

    # merge channels and convert back to rgb
    hsv_stretched = cv2.merge([h, s, v_stretched])
    img_stretched = cv2.cvtColor(hsv_stretched, cv2.COLOR_HSV2BGR)
    return img_stretched


def _helper(input_path, output_dir):
    print("Processing...", input_path)
    filename = os.path.basename(input_path)
    output_path = output_dir + filename

    img = cv2.imread(input_path)
    img_adjusted = _contrast_stretch(img, 50)
    cv2.imwrite(output_path, img_adjusted)


if __name__ == "__main__":
    dotenv.load_dotenv()
    BASE_NAME = os.getenv("BASE_NAME")
    FILE_NAME = os.getenv("FILE_NAME")
    SOURCE_DIR = f"{BASE_NAME}/{FILE_NAME}"
    OUT_DIR = f"{BASE_NAME}/{FILE_NAME}_contrast-stretch/"
    JOBLIB_N_JOBS = -1

    input_path_list = glob.glob(f"{SOURCE_DIR}/*.png")
    os.makedirs(OUT_DIR, exist_ok=True)

    joblib.Parallel(n_jobs=JOBLIB_N_JOBS)(
        joblib.delayed(_helper)(input_path, OUT_DIR) for input_path in input_path_list
    )
