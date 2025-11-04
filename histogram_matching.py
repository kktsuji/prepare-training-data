import os
from glob import glob
from itertools import product
from typing import List

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

_COLORS = ["b", "g", "r"]
_BIT_DEPTH = 8
_MAX_DIGIT = 2**_BIT_DEPTH - 1
_NUM_BINS = _MAX_DIGIT + 1
_ROI_SIZE = 16
_LOCAL_FLAG = False
_LOCAL_GLOBAL = f"in-{_ROI_SIZE}" if _LOCAL_FLAG else "global"

_FILENAME = "xxx"
_TEMPLATE_DIR = "./out/"
_SOURCE_DIR = f"./in/{_FILENAME}/"
_OUT_DIR = f"./out/histogram-matching_{_LOCAL_GLOBAL}/{_FILENAME}/"
_NPY_PATH = f"{_OUT_DIR}average_template_cdf.npy"
_JOBLIB_N_JOBS = -1


def _calculate_cdf(img: NDArray[np.uint8], num_bins: int = 256):
    hist, _ = np.histogram(img.flatten(), num_bins, [0, num_bins])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()
    return hist, cdf_normalized


def _calculate_average_cdf(
    img_path_list: List[str], colors: List[str], num_bins: int = 256
):
    cdf_sum = np.zeros((num_bins, len(colors)))
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        for c, _ in enumerate(colors):
            _, cdf = _calculate_cdf(img[:, :, c], num_bins)
            cdf_sum[:, c] += cdf
    avg_cdf = cdf_sum / len(img_path_list)
    return avg_cdf


def _calculate_average_local_cdf_helper(img, c, y, x, roi_size, num_bins):
    y_start = y * roi_size
    x_start = x * roi_size
    y_end = min(y_start + roi_size, img.shape[0])
    x_end = min(x_start + roi_size, img.shape[1])
    roi = img[y_start:y_end, x_start:x_end, c]
    _, cdf = _calculate_cdf(roi, num_bins)

    return c, y, x, cdf


def _calculate_average_local_cdf(
    img_path_list: List[str], colors: List[str], roi_size: int, num_bins: int = 256
):
    _SIZE = 512
    img = cv2.imread(img_path_list[0])
    num_rois_h = _SIZE // roi_size
    num_rois_w = _SIZE // roi_size
    cdf_sum = np.zeros((num_bins, len(colors), num_rois_h, num_rois_w))

    for idx, img_path in enumerate(img_path_list):
        if idx % 5 == 0:
            print(f"  - {idx+1}/{len(img_path_list)}")
        img = cv2.imread(img_path)
        if img.shape[0] != _SIZE or img.shape[1] != _SIZE:
            img = cv2.resize(img, (_SIZE, _SIZE), interpolation=cv2.INTER_LINEAR)

        cdf_list = joblib.Parallel(n_jobs=_JOBLIB_N_JOBS)(
            joblib.delayed(_calculate_average_local_cdf_helper)(
                img, c, y, x, roi_size, num_bins
            )
            for c, y, x in product(
                range(len(colors)), range(num_rois_h), range(num_rois_w)
            )
        )

        for c, y, x, cdf in cdf_list:
            cdf_sum[:, c, y, x] += cdf

    return cdf_sum / len(img_path_list)


def _calculate_histogram_matching(
    source: NDArray[np.uint8],
    source_cdf_normalized: NDArray[float],
    template_cdf_normalized: NDArray[float],
    num_bins: int = 256,
):
    mapping = np.zeros(num_bins, dtype=np.uint8)
    for src_pixel_val in range(num_bins):
        diff = np.abs(source_cdf_normalized[src_pixel_val] - template_cdf_normalized)
        mapping[src_pixel_val] = np.argmin(diff)
    matched = mapping[source]
    return mapping, matched


def _save_histogram(
    hist: NDArray[int],
    cdf: NDArray[float],
    title: str,
    output_path: str,
):
    plt.figure()
    plt.title(title)
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.plot(hist, color="blue", label="Histogram")
    plt.plot(cdf, color="red", label="Cumulative Distribution Function")
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def _save_cdf(
    cdf_source: NDArray[float],
    cdf_template: NDArray[float],
    cdf_matching: NDArray[float],
    cdf_mapping: NDArray[float],
    title: str,
    output_path: str,
):
    plt.figure()
    plt.title(title)
    plt.xlabel("Pixel value")
    plt.ylabel("Cumulative Distribution Function")
    plt.plot(cdf_source, color="blue", label="Source CDF")
    plt.plot(cdf_template, color="orange", label="Template CDF", lw=5)
    plt.plot(cdf_matching, color="red", label="Matching CDF")
    plt.plot(cdf_mapping, color="green", label="Mapping CDF")
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def _calculate_histogram_matching_helper(
    img_path,
    average_template_cdf,
):
    print(f"Processing... {img_path}")
    source = cv2.imread(img_path)
    dst = np.zeros_like(source)

    for c, _ in enumerate(_COLORS):
        source_ch = source[:, :, c]
        _, cdf_source = _calculate_cdf(source_ch, _NUM_BINS)

        if _LOCAL_FLAG:
            num_rois_h = source.shape[0] // _ROI_SIZE
            num_rois_w = source.shape[1] // _ROI_SIZE
            for y, x in product(range(num_rois_h), range(num_rois_w)):
                y_start = y * _ROI_SIZE
                x_start = x * _ROI_SIZE
                y_end = min(y_start + _ROI_SIZE, source.shape[0])
                x_end = min(x_start + _ROI_SIZE, source.shape[1])
                roi = source_ch[y_start:y_end, x_start:x_end]

                _, matched_ch_roi = _calculate_histogram_matching(
                    roi,
                    cdf_source,
                    average_template_cdf[:, c, y, x],
                    _NUM_BINS,
                )
                dst[y_start:y_end, x_start:x_end, c] = matched_ch_roi
        else:
            _, matched_ch = _calculate_histogram_matching(
                source_ch, cdf_source, average_template_cdf[:, c], _NUM_BINS
            )
            dst[:, :, c] = matched_ch

    filename = os.path.basename(img_path)
    cv2.imwrite(f"{_OUT_DIR}{filename}", dst)


if __name__ == "__main__":
    os.makedirs(_OUT_DIR, exist_ok=True)
    template_path_list = glob(f"{_TEMPLATE_DIR}*.png")
    source_path_list = glob(f"{_SOURCE_DIR}*.png")

    print(f"Number of template images: {len(template_path_list)}")
    if _NPY_PATH != None and os.path.exists(_NPY_PATH):
        print(f"Loading average CDF from {_NPY_PATH}...")
        average_template_cdf = np.load(_NPY_PATH)
    else:
        print("Calculating average CDF from template images...")
        if _LOCAL_FLAG:
            print("  - Using local histogram matching")
            average_template_cdf = _calculate_average_local_cdf(
                template_path_list, _COLORS, _ROI_SIZE, _NUM_BINS
            )
        else:
            print("  - Using global histogram matching")
            average_template_cdf = _calculate_average_cdf(
                template_path_list, _COLORS, _NUM_BINS
            )
        np.save(f"{_OUT_DIR}average_template_cdf.npy", average_template_cdf)

    joblib.Parallel(n_jobs=_JOBLIB_N_JOBS)(
        joblib.delayed(_calculate_histogram_matching_helper)(
            img_path, average_template_cdf
        )
        for img_path in source_path_list
    )
