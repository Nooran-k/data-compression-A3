import math
import numpy as np
from PIL import Image
from codebook import load_codebook

def load_bits(path):
    with open(path, "rb") as file:
        data = file.read()

    bits = "".join(format(byte, "08b") for byte in data)
    return bits


def bits_to_indices(bits, bits_per_index, number_of_indices):
    indices = []

    for i in range(number_of_indices):
        start = i * bits_per_index
        end = start + bits_per_index

        index_bits = bits[start:end]

        if len(index_bits) != bits_per_index:
            raise ValueError("Compressed file is incomplete.")

        index = int(index_bits, 2)
        indices.append(index)

    return np.array(indices, dtype=np.int64)


def reconstruct_image(indices, codebook, padded_shape, block_size):
    bh, bw = block_size
    padded_height, padded_width = padded_shape

    blocks_per_row = padded_width // bw
    blocks_per_column = padded_height // bh

    expected_blocks = blocks_per_row * blocks_per_column

    if len(indices) != expected_blocks:
        raise ValueError("Invalid number of indices.")

    if np.any(indices >= len(codebook)):
        raise ValueError("Invalid codebook index.")

    selected_blocks = codebook[indices]

    image = selected_blocks.reshape(
        padded_height,
        padded_width
    )

    return np.clip(image, 0, 255).astype(np.uint8)


def decompress_image(
    codebook_path="codebook.npz",
    compressed_path="compressed.bin",
    output_path="decompressed.png"
):

    # 1. Load codebook information
    info = load_codebook(codebook_path)

    codebook = info["codebook"]
    block_size = info["block_size"]
    original_shape = info["original_shape"]
    padded_shape = info["padded_shape"]

    # 2. Calculate bits per index
    bits_per_index = max(
        1,
        math.ceil(math.log2(len(codebook)))
    )

    # 3. Calculate expected number of blocks
    bh, bw = block_size
    padded_height, padded_width = padded_shape

    number_of_indices = (
        (padded_height // bh) *
        (padded_width // bw)
    )

    # 4. Read binary stream
    bits = load_bits(compressed_path)

    # 5. Convert binary stream into indices
    indices = bits_to_indices(
        bits,
        bits_per_index,
        number_of_indices
    )

    # 6. Reconstruct padded image
    reconstructed = reconstruct_image(
        indices,
        codebook,
        padded_shape,
        block_size
    )

    # 7. Remove padding
    original_height, original_width = original_shape

    reconstructed = reconstructed[
        :original_height,
        :original_width
    ]

    # 8. Save reconstructed image
    Image.fromarray(reconstructed).save(output_path)

    print("Decompression completed successfully!")
    print("Output image saved to:", output_path)


if __name__ == "__main__":

    codebook_path = input("Codebook file path: ")
    compressed_path = input("Compressed file path: ")
    output_path = input("Output image path: ")

    decompress_image(
        codebook_path,
        compressed_path,
        output_path
    )
