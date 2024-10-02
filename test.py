import re
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def load_class_names(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        class_names = data["UAV_SOD"]["CATEGORY_ID_TO_NAME"]
    return class_names

def parse_annotations(annotation_path, class_names):
    boxes = []
    with open(annotation_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            x_min, y_min, x_max, y_max, class_code = map(int, parts[:5])
            class_name = class_names[str(class_code)]
            mask_string = re.search(r'\[\((.*?)\)\]', line).group(1) # type: ignore
            mask_coords = [tuple(map(int, pair.split(','))) for pair in mask_string.split('), (')]
            # Calculate bounding box from mask
            xs, ys = zip(*mask_coords)
            mask_x_min, mask_x_max = min(xs), max(xs)
            mask_y_min, mask_y_max = min(ys), max(ys)
            boxes.append((x_min, y_min, x_max, y_max, class_name, mask_coords, (mask_x_min, mask_y_min, mask_x_max, mask_y_max)))
    return boxes

def plot_image_with_boxes_and_masks(image_path, boxes):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
        
        
    for (x_min, y_min, x_max, y_max, class_name, mask_coords, mask_bbox) in boxes:
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        draw.rectangle(mask_bbox, outline="black", width=1)
        text_position = (x_min, y_min - 10)
        draw.text(text_position, class_name, fill="red", font=font)
        draw.polygon(mask_coords, outline="black")
    
    image = np.array(image)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.axis('off')
    ax.text(0.01, 1, "Ground Bounding Boxes", transform=ax.transAxes, fontsize=10, color='red', verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', edgecolor='red'))
    ax.text(0.01, 0.97, "Ground Truth Masks", transform=ax.transAxes, fontsize=10, color='black', verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', edgecolor='black'))
    plt.savefig("uav_ground_truth.jpg")
    plt.show()

image_path = '1.jpg'
annotation_path = '1.txt'
json_file = 'src/code_map.json'
class_names = load_class_names(json_file)
boxes = parse_annotations(annotation_path, class_names)
plot_image_with_boxes_and_masks(image_path, boxes)
