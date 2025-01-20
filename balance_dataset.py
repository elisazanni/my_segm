import os
import cv2
import numpy as np
import json
import sys


def analyze_images(input_folder, output_file):
    color_classes = {
        (0, 0, 0): 'Background',
        (0, 0, 200): 'Tumor',
        (50, 50, 50): 'Necrosis',
        (255, 255, 0): 'Tissue',
        (0, 128, 0): 'Cirrhotic tissue',
        (204, 102, 77): 'Exogenous material'
    }

    color_stats = {color: {'pixels': 0, 'occurrences': 0} for color in color_classes}

    total_pixels = 0
    total_images = 0
    unknown_colors = set()

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is not None:
                total_pixels += image.shape[0] * image.shape[1]
                total_images += 1

                # Check for each pixel and classify by color
                for row in image:
                    for pixel in row:
                        color_tuple = tuple(pixel)
                        if color_tuple in color_classes:
                            color_stats[color_tuple]['pixels'] += 1
                        else:
                            unknown_colors.add(color_tuple)

    # Check if any unknown colors were encountered
    if unknown_colors:
        print("Error: The following colors are not defined in the color_classes list:")
        for color in unknown_colors:
            print(color)
    else:
        print("All colors are valid.")

    # Calculate percentages
    for color in color_stats:
        color_stats[color]['percentage'] = (color_stats[color][
                                                'pixels'] / total_pixels) * 100 if total_pixels > 0 else 0

    # Save results to output file
    with open(output_file, 'w') as file:
        json.dump(color_stats, file, indent=4)

    print(f"Analysis complete. Results saved in {output_file}")


# Usage
input_folder = sys.argv[1]
output_file = 'color_analysis_results.json'
analyze_images(input_folder, output_file)
