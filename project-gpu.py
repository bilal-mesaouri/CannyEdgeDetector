import argparse
import numpy as np
from PIL import Image
from numba import cuda
import math

# Define global variables for thread block size
TB_WIDTH = 16
TB_HEIGHT = 16

def bw_kernel(image_array):
    global TB_WIDTH, TB_HEIGHT
    # Use TB_WIDTH and TB_HEIGHT inside the kernel operation

    # Define the grayscale kernel
    bw_kernel = np.array([0.2989, 0.5870, 0.1140])

    # Define the function to apply grayscale kernel to all pixels in parallel
    @cuda.jit
    def apply_grayscale_kernel(image_array, grayscale_image_array, bw_kernel):
        i, j = cuda.grid(2)
        if i < image_array.shape[0] and j < image_array.shape[1]:
            # Calculate the dot product manually for each channel
            grayscale_value = image_array[i, j, 0] * bw_kernel[0] + \
                            image_array[i, j, 1] * bw_kernel[1] + \
                            image_array[i, j, 2] * bw_kernel[2]
            # Set the grayscale value for one channel
            grayscale_image_array[i, j] = grayscale_value

    # Calculate grid and block dimensions
    threadsperblock = (TB_WIDTH, TB_HEIGHT)
    blockspergrid_x = (image_array.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (image_array.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Allocate memory for grayscale image
    grayscale_image_array = np.empty((image_array.shape[0], image_array.shape[1]), dtype=np.float64)

    # Launch the kernel
    apply_grayscale_kernel[blockspergrid, threadsperblock](image_array, grayscale_image_array, bw_kernel)

    # Normalize the grayscale image array
    return np.uint8(grayscale_image_array)

def gauss_kernel(image_array):
    global TB_WIDTH, TB_HEIGHT
    # Use TB_WIDTH and TB_HEIGHT inside the kernel operation

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
    threadsperblock = (TB_WIDTH, TB_HEIGHT)
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
    return filtered_image_array

def sobel_kernel(image_array):
    global TB_WIDTH, TB_HEIGHT
    print("TB : ",TB_HEIGHT, TB_WIDTH)
    # Use TB_WIDTH and TB_HEIGHT inside the kernel operation

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
                output_image[i + 1, j + 1] = math.sqrt(gx**2 + gy**2)

    output = np.zeros_like(image_array)
    # Create CUDA device array for the image
    d_image = cuda.to_device(image_array)
    d_output = cuda.to_device(output)

    # Define block and grid dimensions
    threadsperblock = (TB_WIDTH, TB_HEIGHT)
    blockspergrid_x = (image_array.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (image_array.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Apply Sobel filter using CUDA
    sobel_filter[blockspergrid, threadsperblock](d_image, d_output, sobel_x, sobel_y)

    # Copy the result back to the CPU
    d_output.copy_to_host(output)

    # Normalize the grayscale image array
    return np.uint8(output)
    

def threshold_kernel(image, low_thresh, high_thresh):
    global TB_WIDTH, TB_HEIGHT
    # Use TB_WIDTH and TB_HEIGHT inside the kernel operation

    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # Create output array
    output_edges = np.zeros_like(image)

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
    threads_per_block = (TB_WIDTH, TB_HEIGHT)
    blocks_per_grid_x = (image.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (image.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Prepare data on the device
    edges_device = cuda.to_device(image)
    output_device = cuda.to_device(output_edges)

    # Launch the kernel
    hysteresis_kernel[blocks_per_grid, threads_per_block](edges_device, output_device, low_thresh, high_thresh)

    # Copy back to host and return the result
    output_edges = output_device.copy_to_host()
    return output_edges

def main():
    global TB_WIDTH, TB_HEIGHT
    parser = argparse.ArgumentParser(description='Perform various image processing operations.')
    parser.add_argument('inputImage', help='the source image')
    parser.add_argument('outputImage', help='the destination image')
    parser.add_argument('--tb', type=int, help='optional size of a thread block for all operations')
    parser.add_argument('--bw', action='store_true', help='perform only the bw_kernel')
    parser.add_argument('--gauss', action='store_true', help='perform the bw_kernel and the gauss_kernel')
    parser.add_argument('--sobel', action='store_true', help='perform all kernels up to sobel_kernel and write to disk the magnitude of each pixel')
    parser.add_argument('--threshold', action='store_true', help='perform all kernels up to threshold_kernel')
    parser.add_argument('--hysteresis', action='store_true', help='perform hysteresis_kernel')

    args = parser.parse_args()

    # Load the image
    image = Image.open(args.inputImage)
    image_array = np.array(image)

    if args.tb:
        TB_WIDTH = args.tb
        TB_HEIGHT = args.tb
    if args.bw:
        bw_image_array = bw_kernel(image_array)
        grayscale_image = Image.fromarray(bw_image_array)
        grayscale_image.save(args.outputImage)
        return
    
    if args.gauss:
        bw_image_array = bw_kernel(image_array)
        gauss_image_array = gauss_kernel(bw_image_array)
        gauss_image = Image.fromarray(gauss_image_array)
        gauss_image.save(args.outputImage)
        return
    
    if args.sobel:

        bw_image_array = bw_kernel(image_array)
        gauss_image_array = gauss_kernel(bw_image_array)
        sobel_image_array = sobel_kernel(gauss_image_array)
        sobel_image = Image.fromarray(sobel_image_array)
        sobel_image.save(args.outputImage)
        return
    
    if args.threshold:

        bw_image_array = bw_kernel(image_array)
        gauss_image_array = gauss_kernel(bw_image_array)
        sobel_image_array = sobel_kernel(gauss_image_array)
        # Threshold the image
        thresholded_image = threshold_kernel(sobel_image_array, low_thresh=75, high_thresh=150)
        thresholded_img = Image.fromarray(thresholded_image)
        thresholded_img.save(args.outputImage)
        return
    
    if not (args.bw and args.gauss and args.sobel and args.threshold):
        # If no kernel-specific argument is provided, perform all kernels
        bw_image_array = bw_kernel(image_array)
        gauss_image_array = gauss_kernel(bw_image_array)
        sobel_image_array = sobel_kernel(gauss_image_array)
        thresholded_image = threshold_kernel(sobel_image_array, low_thresh=75, high_thresh=150)
        final_image = Image.fromarray(thresholded_image)
        final_image.save(args.outputImage)
        return

if __name__ == "__main__":
    main()