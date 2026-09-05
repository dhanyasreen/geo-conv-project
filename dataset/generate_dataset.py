"""
generate_dataset.py
--------------------
Generates a synthetic single-channel "geographic feature map" that mimics
a satellite / aerial image containing objects at three very different
spatial scales:

    1. Small scale   -> isolated bright dots      (e.g. individual buildings)
    2. Medium scale  -> thin elongated strips      (e.g. roads / rivers)
    3. Large scale    -> big soft blobs             (e.g. lakes / forest patches)

The synthetic image is used (instead of a downloaded satellite image) so the
project has no external data dependency and is fully reproducible.
The array is also exported as a CSV so it can be inspected like a normal
tabular dataset (customer_data.csv equivalent for this project).
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

SIZE = 128


def make_geo_image(size=SIZE):
    img = np.zeros((size, size), dtype=np.float32)

    # ---- Large-scale feature: a big soft circular blob (lake) ----
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy, r = size * 0.7, size * 0.65, size * 0.22
    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2)))
    img += blob * 0.9

    cx2, cy2, r2 = size * 0.25, size * 0.25, size * 0.15
    blob2 = np.exp(-(((xx - cx2) ** 2 + (yy - cy2) ** 2) / (2 * r2 ** 2)))
    img += blob2 * 0.7

    # ---- Medium-scale feature: thin diagonal + straight strips (roads) ----
    for offset in range(-2, 3):
        idx = np.clip(np.arange(size) + offset, 0, size - 1)
        img[idx, np.arange(size)] += 0.5          # diagonal road
    road_row = size // 2
    img[road_row - 1:road_row + 1, :] += 0.5        # horizontal road

    # ---- Small-scale features: isolated bright dots (buildings) ----
    n_dots = 60
    xs = np.random.randint(4, size - 4, n_dots)
    ys = np.random.randint(4, size - 4, n_dots)
    for x, y in zip(xs, ys):
        img[y - 1:y + 2, x - 1:x + 2] += 0.8

    # normalize to [0, 1]
    img = np.clip(img, 0, None)
    img = img / img.max()
    return img


if __name__ == "__main__":
    img = make_geo_image()
    out_dir = os.path.dirname(__file__)
    np.save(os.path.join(out_dir, "geo_image.npy"), img)
    pd.DataFrame(img).to_csv(os.path.join(out_dir, "geo_image.csv"),
                              index=False, header=False)
    print("Synthetic geo-image generated:", img.shape)
    print("Saved to geo_image.npy and geo_image.csv")
