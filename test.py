import pandas as pd 
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


data = pd.read_csv('1.txt', sep=",", header=None)
data.columns = ["x_min", "y_min", "x_max", "y_max", "class_code"]

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
        draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
        # Optionally add text (class_code) above the bounding box
        draw.text((x_min, y_min), str(class_code), fill='yellow')
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()



draw_bounding_boxes('1.jpg', data)