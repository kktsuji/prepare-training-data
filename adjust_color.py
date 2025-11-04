import glob
import os

import cv2
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


def _adjust_color(img):
    gain = 0.6

    dst = img.copy().astype(np.float32)
    dst[:, :, 2] *= gain

    print("  -", img[255, 255, 2], dst[255, 255, 2])

    return dst


def _apply_gamma_correction(img, gamma=2.2):
    img_normalized = img.astype(np.float32) / 255.0
    img_corrected = np.power(img_normalized, 1 / gamma)
    return (img_corrected * 255).astype(np.uint8)


if __name__ == "__main__":
    pass
