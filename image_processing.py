import numpy as np
from PIL import Image

def load_image(image_path):
    image = Image.open(image_path).convert("L")
    img = np.array(image, dtype=np.uint8)

    if img.size == 0:
        raise ValueError("The image is empty.")

    return img, img.shape

def pad_image(img, block_size):
    bh, bw = block_size
    if bh <= 0 or bw <= 0:
        raise ValueError("Block dimensions must be positive.")

    height, width = img.shape
    pad_h = (bh - height % bh) % bh
    pad_w = (bw - width % bw) % bw

    padded_img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")
    return padded_img, (pad_h, pad_w)

def extract_blocks(padded_img, block_size):
    bh, bw = block_size
    height, width = padded_img.shape

    if height % bh != 0 or width % bw != 0:
        raise ValueError("Image dimensions must be divisible by block size.")

    blocks = []
    for i in range(0, height, bh):
        for j in range(0, width, bw):
            block = padded_img[i:i + bh, j:j + bw]
            blocks.append(block.flatten())

    return np.array(blocks, dtype=np.float32)