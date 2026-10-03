"""
preprocess_utils.py
-------------------
Image pre-processing pipeline for diabetic-retinopathy fundus photographs.

Written as a stand-alone module so the exact same code is used for
  (1) training-time caching,
  (2) evaluation / Grad-CAM display, and
  (3) the live demo / any later deployment.
Every function takes and returns RGB uint8 numpy arrays.
"""
from collections import OrderedDict

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def load_rgb(path):
    """Read an image file from disk and return it as an RGB uint8 array."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def crop_fundus(img, tol=10, min_pixels=5):
    """
    Remove the black background around the retina and pad the result to a
    square so that resizing does not distort the (circular) fundus.

    tol         : grey level above which a pixel counts as "retina"
    min_pixels  : a row/column must contain this many bright pixels to be kept
                  (protects against isolated sensor-noise pixels)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.mean() < 0.02:            # almost-black image: do not crop
        crop = img
    else:
        rows = np.where(mask.sum(axis=1) >= min_pixels)[0]
        cols = np.where(mask.sum(axis=0) >= min_pixels)[0]
        if len(rows) == 0 or len(cols) == 0:
            crop = img
        else:
            crop = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = crop.shape[:2]
    side = max(h, w)
    out = np.zeros((side, side, 3), dtype=np.uint8)          # black square canvas
    y0, x0 = (side - h) // 2, (side - w) // 2
    out[y0:y0 + h, x0:x0 + w] = crop                          # centre the retina
    return out


def resize_square(img, size):
    """Resize to size x size. INTER_AREA is the right choice when shrinking."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def circle_mask(size, radius_frac=0.48):
    """Binary mask (float 0/1, HxW) of the circular retina region."""
    m = np.zeros((size, size), dtype=np.float32)
    cv2.circle(m, (size // 2, size // 2), int(size * radius_frac), 1.0, -1)
    return m


# ----------------------------------------------------------------------------
# Enhancement techniques
# ----------------------------------------------------------------------------
def denoise(img, ksize=3, sigma=0.8):
    """Light Gaussian smoothing (noise removal). Kept small so that tiny
    micro-aneurysms are not wiped out."""
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_clahe(img, clip_limit=2.0, tile=8):
    """
    Contrast Limited Adaptive Histogram Equalisation on the L (lightness)
    channel of LAB colour space. Boosts local contrast without shifting colours.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def unsharp_mask(img, sigma=3.0, amount=0.5):
    """Classic edge enhancement: img + amount * (img - blurred_img)."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def ben_graham(img, sigma_frac=1.0 / 30.0):
    """
    Local-average colour subtraction (popularised by B. Graham in the 2015
    Kaggle DR competition). Subtracting a heavily blurred copy removes
    illumination / colour differences between cameras and acts as a high-pass
    (edge-enhancing) filter that makes vessels, haemorrhages and exudates stand
    out. The outside of the retina is set to neutral grey (128).
    """
    size = img.shape[0]
    blur = cv2.GaussianBlur(img, (0, 0), size * sigma_frac)
    out = cv2.addWeighted(img, 4, blur, -4, 128)
    m = circle_mask(size)[..., None]
    out = out.astype(np.float32) * m + 128.0 * (1.0 - m)
    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Full pipelines
# ----------------------------------------------------------------------------
def preprocess_steps(img_rgb, size=256):
    """Return every intermediate result (used for the explanatory figure)."""
    steps = OrderedDict()
    steps["1. Original"] = img_rgb
    cropped = crop_fundus(img_rgb)
    steps["2. Cropped + squared"] = cropped
    resized = resize_square(cropped, size)
    steps["3. Resized"] = resized
    steps["4a. Denoise + CLAHE"] = apply_clahe(denoise(resized))
    steps["4b. Denoise + unsharp"] = unsharp_mask(denoise(resized))
    steps["4c. Ben Graham (final)"] = ben_graham(resized)
    return steps


def preprocess_rgb(img_rgb, size=256, mode="ben"):
    """
    Pipeline used for training and inference.

    mode = "ben"   : crop -> resize -> Ben Graham colour normalisation (default)
    mode = "clahe" : crop -> resize -> denoise -> CLAHE
    mode = "none"  : crop -> resize only (baseline)
    """
    x = resize_square(crop_fundus(img_rgb), size)
    if mode == "ben":
        return ben_graham(x)
    if mode == "clahe":
        return apply_clahe(denoise(x))
    if mode == "none":
        return x
    raise ValueError(f"Unknown preprocessing mode: {mode}")


def load_display(path, size=256):
    """Cropped + resized image WITHOUT enhancement (natural colours) for plots."""
    return resize_square(crop_fundus(load_rgb(path)), size)


def load_and_preprocess(path, size=256, mode="ben"):
    """Disk -> model-ready uint8 image."""
    return preprocess_rgb(load_rgb(path), size, mode)


# ----------------------------------------------------------------------------
# Simple image-quality metrics (used to justify the pipeline quantitatively)
# ----------------------------------------------------------------------------
def fundus_quality(img, radius_frac=0.40):
    """
    Return (contrast, sharpness) measured on the green channel inside the retina.
    The green channel gives the best vessel/lesion contrast in fundus images.
      contrast  : standard deviation of green intensities
      sharpness : variance of the Laplacian (higher = more edge detail)
    """
    g = img[..., 1].astype(np.float32)
    m = circle_mask(img.shape[0], radius_frac) > 0
    lap = cv2.Laplacian(g, cv2.CV_32F)
    return float(g[m].std()), float(lap[m].var())
