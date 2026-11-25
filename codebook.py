import numpy as np
from image_processing import load_image, pad_image, extract_blocks

def assign_nearest(blocks, codebook):
    distances = np.sum((blocks[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1)

def calculate_error(blocks, codebook):
    labels = assign_nearest(blocks, codebook)
    differences = blocks - codebook[labels]
    return np.mean(np.sum(differences ** 2, axis=1))

def split_codebook(codebook, split_index, epsilon):
    centroid = codebook[split_index]
    delta = epsilon * np.maximum(np.abs(centroid), 1.0)

    new_centroids = np.array([centroid + delta, centroid - delta], dtype=np.float32)
    new_codebook = np.delete(codebook, split_index, axis=0)
    new_codebook = np.concatenate((new_codebook, new_centroids), axis=0)

    return new_codebook

def refine_codebook(blocks, codebook, max_iter=50):
    previous_error = float("inf")

    for _ in range(max_iter):
        labels = assign_nearest(blocks, codebook)
        differences = blocks - codebook[labels]
        distances = np.sum(differences ** 2, axis=1)

        new_codebook = np.zeros_like(codebook)

        for i in range(len(codebook)):
            cluster = blocks[labels == i]
            if len(cluster) > 0:
                new_codebook[i] = np.mean(cluster, axis=0)
            else:
                farthest_index = np.argmax(distances)
                new_codebook[i] = blocks[farthest_index]
                distances[farthest_index] = -1

        codebook = new_codebook
        error = calculate_error(blocks, codebook)

        if abs(previous_error - error) < 1e-6:
            break
        previous_error = error

    return codebook

def generate_codebook(blocks, codebook_size, epsilon=0.01, max_iter=50):
    if codebook_size <= 0:
        raise ValueError("Codebook size must be positive.")
    if len(blocks) == 0:
        raise ValueError("No blocks available.")
    if codebook_size > len(blocks):
        raise ValueError("Codebook size cannot exceed number of blocks.")
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")

    codebook = np.mean(blocks, axis=0, keepdims=True)

    while len(codebook) < codebook_size:
        labels = assign_nearest(blocks, codebook)
        differences = blocks - codebook[labels]
        distances = np.sum(differences ** 2, axis=1)

        cluster_errors = np.zeros(len(codebook))
        for i in range(len(codebook)):
            cluster_errors[i] = np.sum(distances[labels == i])

        split_index = np.argmax(cluster_errors)
        codebook = split_codebook(codebook, split_index, epsilon)
        codebook = refine_codebook(blocks, codebook, max_iter)

    return codebook

def save_codebook(path, codebook, block_size, original_shape, padded_shape, padding):
    if not path.endswith(".npz"):
        path += ".npz"

    np.savez(
        path,
        codebook=codebook,
        block_size=np.array(block_size),
        num_vectors=len(codebook),
        original_shape=np.array(original_shape),
        padded_shape=np.array(padded_shape),
        padding=np.array(padding),
        is_grayscale=np.array(True),
    )
    return path

def load_codebook(path):
    with np.load(path, allow_pickle=False) as data:
        information = {
            "codebook": data["codebook"],
            "block_size": tuple(data["block_size"]),
            "num_vectors": int(data["num_vectors"]),
            "original_shape": tuple(data["original_shape"]),
            "padded_shape": tuple(data["padded_shape"]),
            "padding": tuple(data["padding"]),
            "is_grayscale": bool(data["is_grayscale"]),
        }
    return information

def build_codebook(image_path, block_size, codebook_size, codebook_path="codebook.npz"):
    if isinstance(block_size, int):
        block_size = (block_size, block_size)

    if len(block_size) != 2:
        raise ValueError("Block size must have two dimensions.")

    bh, bw = block_size
    if bh <= 0 or bw <= 0:
        raise ValueError("Block dimensions must be positive.")

    img, original_shape = load_image(image_path)
    padded_img, padding = pad_image(img, block_size)
    padded_shape = padded_img.shape
    blocks = extract_blocks(padded_img, block_size)
    codebook = generate_codebook(blocks, codebook_size)

    codebook_path = save_codebook(codebook_path, codebook, block_size, original_shape, padded_shape, padding)

    return {
        "blocks": blocks,
        "codebook": codebook,
        "codebook_path": codebook_path,
        "block_size": block_size,
        "original_shape": original_shape,
        "padded_shape": padded_shape,
        "padding": padding,
    }