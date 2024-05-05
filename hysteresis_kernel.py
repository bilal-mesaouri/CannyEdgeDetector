from numba import cuda
import numpy as np
from PIL import Image

# Load your thresholded image data
edge_image_path = "./edges_detecteddd.jpg"  # Adjust with your correct path
edge_image = Image.open(edge_image_path)
edges = np.array(edge_image, dtype=np.uint8)

# Create output array
output_edges = np.zeros_like(edges)

# Define the hysteresis CUDA kernel
@cuda.jit
def hysteresis_kernel(input_edges, output_edges, low_thresh, high_thresh):
    i, j = cuda.grid(2)
    if i < input_edges.shape[0] and j < input_edges.shape[1]:
        current = input_edges[i, j]
        if current >= high_thresh:
            output_edges[i, j] = 255
        elif current <= low_thresh:
            output_edges[i, j] = 0
        else:  # Potential weak edge
            # Check 8-connected neighborhood
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue  # Skip the center pixel
                    ni, nj = i + di, j + dj
                    if 0 <= ni < input_edges.shape[0] and 0 <= nj < input_edges.shape[1]:
                        if input_edges[ni, nj] >= high_thresh:
                            output_edges[i, j] = 255
                            break
                if output_edges[i, j] == 255:
                    break

# Configuration for kernel execution
threads_per_block = (16, 16)
blocks_per_grid_x = (edges.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (edges.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# Define thresholds
low_thresh = 75  # Adjust based on your image characteristics
high_thresh = 150  # Adjust based on your image characteristics

# Prepare data on the device
edges_device = cuda.to_device(edges)
output_device = cuda.to_device(output_edges)

# Launch the kernel
hysteresis_kernel[blocks_per_grid, threads_per_block](edges_device, output_device, low_thresh, high_thresh)

# Copy back to host and save/display the result
output_edges = output_device.copy_to_host()
result_image = Image.fromarray(output_edges)
result_image.save("./hysteresis_output.jpg")
