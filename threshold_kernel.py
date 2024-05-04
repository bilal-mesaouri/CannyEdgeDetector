import numpy as np
from numba import cuda
from PIL import Image
import matplotlib.pyplot as plt

# Load the Sobel processed image
image = Image.open("./Sobelvalve.jpg")
image_array = np.array(image.convert('L'))  # Convert to grayscale
edge_magnitude = np.sqrt(image_array.astype(np.float32))  # Example calculation

# Check the range of edge magnitudes
print("Max magnitude:", np.max(edge_magnitude))
print("Min magnitude:", np.min(edge_magnitude))

@cuda.jit
def threshold_kernel(edge_magnitude, edges_output, low_thresh, high_thresh):
    i, j = cuda.grid(2)
    if i < edge_magnitude.shape[0] and j < edge_magnitude.shape[1]:
        if edge_magnitude[i, j] > high_thresh:
            edges_output[i, j] = 255  # Mark as strong edge
        elif edge_magnitude[i, j] > low_thresh:
            edges_output[i, j] = 100  # Mark as weak edge
        else:
            edges_output[i, j] = 0  # Mark as non-edge

edges_output = np.zeros_like(edge_magnitude, dtype=np.uint8)

# Set thresholds
# Using a higher percentage for a more aggressive thresholding based on the max magnitude
low_threshold = max(5, np.max(edge_magnitude) * 0.3)  # Ensuring a minimum threshold of 5
high_threshold = max(10, np.max(edge_magnitude) * 0.5)  # Ensuring a minimum threshold of 10
# Define block and grid dimensions
threads_per_block = (16, 16)
blocks_per_grid_x = (edge_magnitude.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (edge_magnitude.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# Launch the kernel
d_edge_magnitude = cuda.to_device(edge_magnitude)
d_edges_output = cuda.to_device(edges_output)
threshold_kernel[blocks_per_grid, threads_per_block](d_edge_magnitude, d_edges_output, low_threshold, high_threshold)

# Copy the result back to the host
d_edges_output.copy_to_host(edges_output)

# Save and show output for debugging
output_image = Image.fromarray(edges_output)
output_image.save("./edges_detecteddd.jpg")

