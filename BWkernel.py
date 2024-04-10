import numpy as np
from PIL import Image
from numba import cuda

# Load the image
image = Image.open("./valveFiltered.jpg")
image_array = np.array(image)

# Define the grayscale kernel
bw_kernel = np.array([0.2989, 0.5870, 0.1140])

# Define the function to apply grayscale kernel to all pixels in parallel
@cuda.jit
def apply_grayscale_kernel(image_array, grayscale_image_array, bw_kernel):
    i, j = cuda.grid(2)
    if i < image_array.shape[0] and j < image_array.shape[1]:
        # Calculate the dot product manually
        grayscale_value = image_array[i, j, 0] * bw_kernel[0] + \
                         image_array[i, j, 1] * bw_kernel[1] + \
                         image_array[i, j, 2] * bw_kernel[2]
        grayscale_image_array[i, j] = grayscale_value

# Calculate grid and block dimensions
threadsperblock = (16, 16)
blockspergrid_x = (image_array.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (image_array.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Allocate memory for grayscale image
grayscale_image_array = np.empty_like(image_array[:, :, 0], dtype=np.float64)

# Launch the kernel
apply_grayscale_kernel[blockspergrid, threadsperblock](image_array, grayscale_image_array, bw_kernel)

# Normalize the grayscale image array
grayscale_image_array = np.uint8(grayscale_image_array)

# Save the grayscale image
grayscale_image = Image.fromarray(grayscale_image_array)
grayscale_image.save("BWvalve.jpg")
