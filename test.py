import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

def read_annotations(file_path):
    objects = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            x_min, y_min, x_max, y_max = map(int, parts[:4])
            class_code = int(parts[4])
            # Parse the mask points
            # mask_points = eval(parts[5])
            objects.append((x_min, y_min, x_max, y_max, class_code))
    return objects

def draw_annotations(image, objects):
    for obj in objects:
        x_min, y_min, x_max, y_max, class_code = obj
        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
        # Draw mask
        # mask_poly = np.array(mask_points, np.int32)
        # cv2.polylines(image, [mask_poly], isClosed=True, color=(0, 255, 0), thickness=2)

def main():
    image_path = '2.jpg'
    annotations_path = '2.txt'
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"The image {image_path} could not be found.")
    
    # Read annotations
    objects = read_annotations(annotations_path)
    
    # Draw annotations on the image
    draw_annotations(image, objects)
    
    # Convert color to RGB (matplotlib default) from BGR (OpenCV default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Plotting
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()



if __name__ == '__main__':
    main()
