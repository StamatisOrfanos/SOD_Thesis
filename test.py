import pandas as pd 
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np


def draw_bounding_boxes(image_path, annotations, normalized=False):
    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    if normalized:
        # Get image dimensions
        width, height = image.size
        
    # Iterate through the bounding box data
    for index, row in annotations.iterrows():
        if normalized:
            # Scale coordinates if they are normalized
            x_min = int(row['x_min'] * width)
            y_min = int(row['y_min'] * height)
            x_max = int(row['x_max'] * width)
            y_max = int(row['y_max'] * height)
        else:
            x_min, y_min, x_max, y_max = int(row['x_min']), int(row['y_min']), int(row['x_max']), int(row['y_max'])
        
        class_code = row['class_code']  # This can be used to label the box
        
        # Draw rectangle on image
        draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=1)
        # Optionally add text (class_code) above the bounding box
        draw.text((x_min, y_min), str(class_code), fill='yellow')
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')  
    plt.show()





def resize_and_pad(image_path, output_size=(600, 600)):
    img = Image.open(image_path)

    # Determine the most common color in the image for padding
    colors = img.getcolors(256 * 256)
    most_common_color = max(colors, key=lambda item: item[0])[1]

    # Resize image while maintaining aspect ratio
    img.thumbnail((output_size[0], output_size[1]), Image.ANTIALIAS)

    # Pad the image if it's not already the target size
    padded_img = ImageOps.pad(img, size=output_size, color=most_common_color)
    padded_img.save(image_path)



def resize_bounding_boxes(image_path, annotation_path, target_size=(600,600)):
    # Load the image to find its original size
    img = Image.open(image_path)
    original_width, original_height = img.size

    # Determine the scale to fit the image within 600x600
    scale = min(target_size[0] / original_width, target_size[1] / original_height)

    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Calculate padding to be added
    pad_width = (target_size[0] - new_width) // 2
    pad_height = (target_size[1] - new_height) // 2

    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    with open(annotation_path, 'w') as file:
        for line in lines:
            x_min, y_min, x_max, y_max, class_code = map(int, line.strip().split(','))
            x_min = int(x_min * scale) + pad_width
            x_max = int(x_max * scale) + pad_width
            y_min = int(y_min * scale) + pad_height
            y_max = int(y_max * scale) + pad_height
            file.write(f'{x_min},{y_min},{x_max},{y_max},{class_code}\n')


# Example usage


# Before
# data = pd.read_csv('2 copy.txt', sep=",", header=None)
# data.columns = ["x_min", "y_min", "x_max", "y_max", "class_code"]
# draw_bounding_boxes('2 copy.jpg', data)




# After
# resize_bounding_boxes('2.jpg', '2.txt')
# resize_and_pad('2.jpg')
data_after = pd.read_csv('2.txt', sep=",", header=None)
data_after.columns = ["x_min", "y_min", "x_max", "y_max", "class_code"]
draw_bounding_boxes("2.jpg", data_after)


