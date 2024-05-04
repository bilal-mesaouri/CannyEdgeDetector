import numpy as np
from numba import cuda
from PIL import Image

# Device function to check neighboring pixels
@cuda.jit(device=True)
def is_connected_to_strong(i, j, edges, strong_val):
    for di in range(-1, 2):
        for dj in range(-1, 2):
            ni, nj = i + di, j + dj
            if ni >= 0 and ni < edges.shape[0] and nj >= 0 and nj < edges.shape[1]:
                if edges[ni, nj] == strong_val:
                    return True
    return False

# Kernel to apply hysteresis
@cuda.jit
def hysteresis_kernel(edge_magnitude, edges_output, low_thresh, high_thresh, strong_val, weak_val):
    i, j = cuda.grid(2)
    if i < edge_magnitude.shape[0] and j < edge_magnitude.shape[1]:
        if edge_magnitude[i, j] > high_thresh:
            edges_output[i, j] = strong_val  # Mark as strong edge
        elif edge_magnitude[i, j] > low_thresh:
            if is_connected_to_strong(i, j, edges_output, strong_val):
                edges_output[i, j] = weak_val  # Mark as weak edge if connected to a strong edge
        else:
            edges_output[i, j] = 0  # Mark as non-edge

def apply_hysteresis(edge_magnitude, low_threshold, high_threshold, strong_val=255, weak_val=75):
    # Initialize output array
    edges_output = np.zeros_like(edge_magnitude, dtype=np.uint8)

    # Define grid and block dimensions
    threads_per_block = (16, 16)
    blocks_per_grid_x = (edge_magnitude.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (edge_magnitude.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Prepare data on device
    d_edge_magnitude = cuda.to_device(edge_magnitude)
    d_edges_output = cuda.to_device(edges_output)

    # Launch kernel
    hysteresis_kernel[blocks_per_grid, threads_per_block](d_edge_magnitude, d_edges_output, low_threshold, high_threshold, strong_val, weak_val)

    # Copy result back to host
    d_edges_output.copy_to_host(edges_output)
    return edges_output

# Example of loading an image and applying hysteresis
if __name__ == "__main__":
    image_path = "./edges_detecteddd.jpg"  # Update with your image path
    image = Image.open(image_path)
    image_array = np.array(image.convert('L'))  # Convert to grayscale
    edge_magnitude = np.abs(np.float32(image_array))  # Simulate an edge magnitude array

    # Apply hysteresis
    edges_output = apply_hysteresis(edge_magnitude, 50, 100)

    # Save or show result
    output_image = Image.fromarray(edges_output)
    output_image.save("hysteresis_output.jpg")  # Save the output image
