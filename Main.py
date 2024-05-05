import numpy as np
from PIL import Image
from numba import cuda

# Load the image
image = Image.open("./BWvalve.jpg")
image_array = np.array(image)

# Define the Gaussian filter kernel
gaussian_filter = np.array([[1, 4, 6, 4, 1],
                             [4, 16, 24, 16, 4],
                             [6, 24, 36, 24, 6],
                             [4, 16, 24, 16, 4],
                             [1, 4, 6, 4, 1]])

# Define the function to apply Gaussian filter to a single channel
@cuda.jit
def apply_gaussian_filter_single_channel(array, filtered_array, gaussian_filter):
    i, j = cuda.grid(2)
    if i < array.shape[0] and j < array.shape[1]:
        # Initialize the sum to the current pixel value
        pixel_sum = array[i, j]
        weight_sum = 1

        # Iterate over the neighbors and apply the filter
        for k in range(-2, 3):
            for l in range(-2, 3):
                if 0 <= i + k < array.shape[0] and 0 <= j + l < array.shape[1]:
                    pixel_sum += array[i + k, j + l] * gaussian_filter[k + 2, l + 2]
                    weight_sum += gaussian_filter[k + 2, l + 2]

        # Normalize the result
        filtered_array[i, j] = pixel_sum / weight_sum

# Define the function to apply Gaussian filter to all channels in parallel
@cuda.jit
def apply_gaussian_filter(image_array, filtered_image_array, gaussian_filter):
    i, j = cuda.grid(2)
    if i < image_array.shape[0] and j < image_array.shape[1]:
        apply_gaussian_filter_single_channel(
            image_array, 
            filtered_image_array,
            gaussian_filter
        )

# Calculate grid and block dimensions
threadsperblock = (16, 16)
blockspergrid_x = (image_array.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (image_array.shape[1] + threadsperblock[1] - 1) // threadsperblock[1] 
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Allocate memory for filtered image
filtered_image_array = np.empty_like(image_array, dtype=np.float64)

# Copy the Gaussian filter to the device
d_gaussian_filter = cuda.to_device(gaussian_filter)

# Launch the kernel
apply_gaussian_filter[blockspergrid, threadsperblock](image_array, filtered_image_array, d_gaussian_filter)

# Normalize the filtered image array
filtered_image_array = np.uint8(filtered_image_array)

# Save the filtered image
filtered_image = Image.fromarray(filtered_image_array)
filtered_image.save("valveFiltered.jpg")
