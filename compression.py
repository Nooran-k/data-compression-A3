import math
import numpy as np
from codebook import build_codebook


def find_indices(blocks, codebook):
    indices = []
    for block in blocks:
        distances = np.sum((codebook - block) ** 2, axis=1)
        indices.append(np.argmin(distances))
    return indices


def indices_to_bits(indices, bits_per_index):
    bits = ""
    for index in indices:
        bits += format(index, "b").zfill(bits_per_index)
    return bits


def save_bits(bits, path):
    while len(bits) % 8 != 0:
        bits += "0"

    data = bytearray()
    for i in range(0, len(bits), 8):
        data.append(int(bits[i:i + 8], 2))

    with open(path, "wb") as file:
        file.write(data)


def compression_ratio(original_shape, codebook, number_of_bits):
    height, width = original_shape
    original_size = height * width * 8
    codebook_size = codebook.size * 8
    return original_size / (number_of_bits + codebook_size)


def compress_image(image_path, block_size, codebook_size,
                   codebook_path="codebook.npz", compressed_path="compressed.bin"):
    result = build_codebook(image_path, block_size, codebook_size, codebook_path)
    blocks = result["blocks"]
    codebook = result["codebook"]

    indices = find_indices(blocks, codebook)

    bits_per_index = max(1, math.ceil(math.log2(len(codebook))))
    bits = indices_to_bits(indices, bits_per_index)

    save_bits(bits, compressed_path)

    ratio = compression_ratio(
        result["original_shape"],
        codebook,
        len(bits)
    )

    return {
        "codebook_path": result["codebook_path"],
        "compressed_path": compressed_path,
        "compression_ratio": ratio
    }
