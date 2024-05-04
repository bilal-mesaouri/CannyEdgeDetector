ùùimport numpy as np
from numba import cuda
from PIL import Image
import math
rawImage = Image.open("./BWvalve.jpg")
image_array = np.array(rawImage)

# Define the Sobel kernel
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

@cuda.jit
def sobel_filter(image, output_image, sobel_x, sobel_y):
      i, j = cuda.grid(2)
    
      if i > 0 and i < image.shape[0] - 1 and j > 0 and j < image.shape[1] - 1:
            # Compute gradient in x direction
            gx = (sobel_x[0, 0] * image[i - 1, j - 1] + sobel_x[0, 1] * image[i - 1, j] + sobel_x[0, 2] * image[i - 1, j + 1] +
                  sobel_x[1, 0] * image[i, j - 1] + sobel_x[1, 1] * image[i, j] + sobel_x[1, 2] * image[i, j + 1] +
                  sobel_x[2, 0] * image[i + 1, j - 1] + sobel_x[2, 1] * image[i + 1, j] + sobel_x[2, 2] * image[i + 1, j + 1])

            # Compute gradient in y direction
            gy = (sobel_y[0, 0] * image[i - 1, j - 1] + sobel_y[0, 1] * image[i - 1, j] + sobel_y[0, 2] * image[i - 1, j + 1] +
                  sobel_y[1, 0] * image[i, j - 1] + sobel_y[1, 1] * image[i, j] + sobel_y[1, 2] * image[i, j + 1] +
                  sobel_y[2, 0] * image[i + 1, j - 1] + sobel_y[2, 1] * image[i + 1, j] + sobel_y[2, 2] * image[i + 1, j + 1])
            output_image[i + 1, j + 1] =math.sqrt(gx**2 + gy**2)                                                                                      
            



output = np.zeros_like(image_array)
# Create CUDA device array for the image
d_image = cuda.to_device(image_array)
d_output = cuda.to_device(output)

print(image_array.shape)


# Define block and grid dimensions
threadsperblock = (16, 16)
blockspergrid_x = (image_array.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (image_array.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Apply Sobel filter using CUDA
sobel_filter[blockspergrid, threadsperblock](d_image, d_output, sobel_x, sobel_y)

# Copy the result back to the CPU

d_output.copy_to_host(output)

# Normalize the grayscale image array
output_image = np.uint8(d_output)
# Print the first 10 pixels of the input image
print("Input Image:")
for i in range(10):
      for j in range(10):
            print(image_array[i, j], end=" ")
      print()

# Print the first 10 pixels of the output image
print("Output Image:")
for i in range(10):
      for j in range(10):
            print(output_image[i, j], end=" ")
      print()

print(output_image.shape)
# Save the grayscale image
sobel_image_ready = Image.fromarray(output_image)
sobel_image_ready.save("Sobelvalve.jpg")

