# Preprocessing pipeline for DIDA (28x28, folders 0-9)
"""
DIDA preprocessing module
contains functions to load and preprocess digit images from folders :
- load_image
- normalize_0_1
- augment_image_pil
- load_dataset_from_folders
"""
__all__ = [
    "load_image",  "normalize_0_1",
     "augment_image_pil", "load_dataset_from_folders"

]

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from sklearn.preprocessing import StandardScaler , MinMaxScaler



# ---------- helper functions ----------
def load_image(path, size=(28,28), to_gray=True):
    img = Image.open(path)
    print(f"Loading image: {path}, original size: {img.size}, mode: {img.mode}")
    if to_gray:
        img = img.convert("L")
        
    if img.size != size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    print(f"Processed image :{path} , resized : {img.size}, mode: {img.mode}")
    return np.array(img, dtype=np.uint8)



def normalize_0_1(img):
    return img.astype(np.float32) / 255.0


# Simple augmentations using PIL transformations (rotate, translate)
# def augment_image_pil(img_pil, rotate=5, translate=2):
#     # small random rotation and translation
#     angle = np.random.uniform(-rotate, rotate)
#     tx = np.random.uniform(-translate, translate)
#     ty = np.random.uniform(-translate, translate)
#     img2 = img_pil.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
#     img2 = img2.transform(img2.size, Image.AFFINE, (1, 0, tx, 0, 1, ty), resample=Image.BILINEAR, fillcolor=0)
#     return img2

# ---------- main loader with preprocessing pipeline ----------
def load_dataset_from_folders(root_dir, 
                              size=(28,28),
                              normalize=True,   # scale to [0,1]
                              augment=False,
                              augment_times=1):
    """
      Load digit images from folders 0–9 and apply a preprocessing pipeline.
 
        Parameters
        ----------
        root_dir : str or Path
            Path containing digit folders named 0–9.
        size : tuple(int, int)


        Pixel Processing
        ----------------
        normalize : bool
            Scale pixels to [0,1].
        standardize : bool
            Apply StandardScaler to flattened images.

        Augmentation
        ------------
        augment : bool
            Enable random rotation/translation augmentation.
        augment_times : int
            How many augmented copies to add per image.

        Returns
        -------
        X : ndarray (n, 28, 28)
            Preprocessed images.
        X_flat : ndarray (n, 784)
            Flattened images for classical ML.
        y : ndarray (n,)
            Digit labels (0–9).

    """

    root = Path(root_dir)
    X = []
    y = []
    for label in range(10):
        folder = root / str(label)
        if not folder.exists(): 
            print("Missing folder:", folder)
            continue
        for f in sorted(folder.iterdir()):
            if not f.is_file(): 
                continue
            arr = load_image(f, size=size)
            # if augment:
            #     # apply augment_times augmentations (plus original)
            #     pil = Image.fromarray(arr)
            #     for _ in range(augment_times):
            #         augp = augment_image_pil(pil)
            #         arr_aug = np.array(augp)
            #         X.append(arr_aug)
            #         y.append(label)
            # X.append(arr)
            # y.append(label)
    X = np.stack(X, axis=0)  # shape (n,28,28)
    y = np.array(y, dtype=np.int64)
    # normalize to [0,1]
    if normalize:
        X = X.astype(np.float32) / 255.0
    # flatten for classical ML
    X_flat = X.reshape((X.shape[0], -1))
    return X, X_flat, y

