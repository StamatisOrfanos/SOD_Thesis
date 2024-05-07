import os, shutil, json, xmltodict
import numpy as np
from tqdm import tqdm
from PIL import ImageOps, Image
import numpy as np
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pycocotools.coco import COCO


# Pre-processing functions for all datasets ------------------------------------------------------------------------

def resize_data(base_path):
    """
    Parameters:
        - base_path (string): Path of directory we want to resize the images and the annotations
    """
    image_path       = os.path.join(base_path, "images")
    annotations_path = os.path.join(base_path, "annotations")
    
    image_files =  [f for f in os.listdir(image_path) if f.endswith('.jpg')]
    for image_file in tqdm(image_files, desc="Resizing the images, annotations and segmentation data to target size (600,600): "):
        image_file_path = os.path.join(image_path, image_file)
        annotation_file = image_file.replace(".jpg", ".txt")
        annotations_file_path = os.path.join(annotations_path, annotation_file)    
        resize_masks_bounding_boxes(image_file_path, annotations_file_path)
        resize_images(image_file_path)
    

def resize_images(image_path, target_size=(600, 600)):
    """
    Parameters:
        - image_path (string): Path of the image we want to resize
        - target_size (tuple, optional): Target size for the new resized image
    """
    img = Image.open(image_path)
    
    colors = img.getcolors(1024 * 1024)  # increase size to handle more colors

    if colors:
        most_common_color = max(colors, key=lambda item: item[0])[1]
    else:
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            colors, counts = np.unique(img_np.reshape(-1, 3), axis=0, return_counts=True)
        else:
            colors, counts = np.unique(img_np.ravel(), return_counts=True)
        most_common_color = colors[counts.argmax()]

    img.thumbnail((target_size[0], target_size[1]), Image.ANTIALIAS)

    # Pad the image if it's not already the target size
    padded_img = ImageOps.pad(img, size=target_size, color=most_common_color)
    padded_img.save(image_path)



def resize_masks_bounding_boxes(image_path, annotation_path, target_size=(600,600)):
    """
    Parameter:
        - image_path (string): Path of the image we want to resize
        - annotation_path (string): Path of the annotation file we want to update
        - target_size (tuple, optional): Target size for the new resized image   
    """
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
            # Resize the annotations to new size
            x_min, y_min, x_max, y_max, class_code, segmentation = map(int, line.strip().split(','))
            x_min = int(x_min * scale) + pad_width
            x_max = int(x_max * scale) + pad_width
            y_min = int(y_min * scale) + pad_height
            y_max = int(y_max * scale) + pad_height
            
            # Resize the annotations 
            resized_segmentation = [(int(x * scale) + pad_width, int(y * scale) + pad_height) for (x, y) in segmentation]
            
            file.write(f'{x_min},{y_min},{x_max},{y_max},{class_code},{resized_segmentation}\n')


def compute_mean_std(images_path, dataset_name):
    """    
    Parameters:
      - images_path (string): The path to the directory containing the images.
      - dataset_name (string): The name of the dataset for which the statistics are computed.
    """
    files = [filename for filename in os.listdir(images_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # If there are more than 2500 images, sample 2500 at random from the dataset
    if len(files) > 2500: files = random.sample(files, 2500)
    pixel_data = []
    
    for filename in files:
        img_path = os.path.join(images_path, filename)
        img = Image.open(img_path)
        img = img.convert('RGB')
        pixels = np.array(img)
        pixel_data.append(pixels)
    
    # Stack all image data
    pixel_data = np.vstack([np.array(image).reshape(-1, 3) for image in pixel_data])
    
    # Compute mean and standard deviation and normalize to 0-1 range
    mean = np.mean(pixel_data, axis=0) / 255 
    std = np.std(pixel_data, axis=0) / 255
    
    # Save to JSON file
    stats = {dataset_name: {'mean': mean.tolist(), 'std': std.tolist()}}            
    with open('src/' + '{}_preprocessing.json'.format(dataset_name), 'w') as f:
        json.dump(stats, f, indent=4)




def plot_random_image_with_annotations(base_path):
    """
    Parameters:
        - base_path (string): Path to the base directory of the dataset
    """
    images_path = os.path.join(base_path, 'train', 'images')
    annotations_path = os.path.join(base_path, 'train', 'annotations')

    # List all images in the images folder and select a random image
    image_files = [f for f in os.listdir(images_path) if f.endswith(".png") or f.endswith("jpg")]
    random_image_name = random.choice(image_files)
    image_path = os.path.join(images_path, random_image_name)
    annotation_path = os.path.join(annotations_path, random_image_name.split(".")[0] + ".txt")

    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Load and parse the annotation file
    with open(annotation_path, 'r') as file:
        annotations = file.readlines()

    # Draw each bounding box
    for annotation in annotations:
        parts = annotation.strip().split(',')
        x_min, y_min, x_max, y_max = map(int, parts[:4])
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()



# COCO dataset ---------------------------------------------------------------------------------------------


def convert_coco_annotations(dataDir):
    # Initialize COCO api for instance annotations
    json_annotation = os.path.join(dataDir, "annotations.json")
    coco = COCO(json_annotation)
    
    # Get all image ids and load annotations
    imgIds = coco.getImgIds()
    images = coco.loadImgs(imgIds)
    
    # Use the same directory to save annotation files
    save_dir = os.path.join(dataDir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for image in images:
        image_filename = image['file_name']
        annotations_ids = coco.getAnnIds(imgIds=image['id'], iscrowd=None)
        annotations = coco.loadAnns(annotations_ids)
        
        # Create a corresponding text file for each image
        txt_filename = os.path.splitext(image_filename)[0] + '.txt'
        with open(os.path.join(save_dir, txt_filename), 'w') as file:
            for annotation in annotations:
                # Extract bounding box and convert to integer x_min, y_min, x_max, y_max
                bbox = annotation["bbox"]
                x_min, y_min, width, height = map(int, map(round, bbox))
                x_max, y_max = x_min + width, y_min + height

                # Category ID
                category_id = annotation['category_id']            
                
                # Segmentation data (default to bounding box data and get segmentation if exist)
                pairs = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

                # Check if segmentation data is valid and in that case use it
                if ("segmentation" in annotation and  annotation["segmentation"] and isinstance(annotation["segmentation"], list) and 
                    len(annotation["segmentation"]) > 0 and  isinstance(annotation["segmentation"][0], list) and  len(annotation["segmentation"][0]) > 0):
                    integer_segmentation = [int(round(x)) for x in annotation["segmentation"][0]]
                    pairs = list(zip(integer_segmentation[::2], integer_segmentation[1::2]))

                formatted_pairs = str(pairs).replace(' ', '')

                file.write(f"{x_min},{y_min},{x_max},{y_max},{category_id},{formatted_pairs}\n")
                
    # Delete the original annotations.json file
    os.remove(json_annotation)



def verify_annotations(dataDir):
    # Paths to the image and annotation directories
    images_dir = os.path.join(dataDir, 'images')
    annotations_dir = os.path.join(dataDir, "annotations")
    
    # List all files in the image and annotation directories
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))}
    annotation_files = {os.path.splitext(f)[0]: f for f in os.listdir(annotations_dir) if f.endswith('.txt')}
    
    # Check for unmatched files
    missing_annotations = [img for img in image_files if img not in annotation_files]
    missing_images = [ann for ann in annotation_files if ann not in image_files]

    # Print results
    if not missing_annotations and not missing_images:
        print("Success: Every image file has a matching annotation file.")
    else:
        if missing_annotations:
            print("Warning: Some images are missing annotations:")
            print(missing_annotations)
        if missing_images:
            print("Warning: Some annotations do not have corresponding images:")
            print(missing_images)
    

# Vis-Drone dateset ----------------------------------------------------------------------------------------

def extract_annotation_values(input_folder):
    """    
    Parameters:
      - input_folder (string): Path of the folder containing the text files.
    """
    # Retrieve all annotations files in the directory and initialize tqdm loop to create the correct format for the annotations
    annotation_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    for file in tqdm(annotation_files, desc="Creating the right annotations format"):
        file_path = os.path.join(input_folder, file)

        with open(file_path, 'r+') as file:
            lines = file.readlines()            
            file.seek(0)
            file.truncate()
            for line in lines:
                values = line.strip().split(',')                
                # Convert (x_min, y_min, width, height) to (x_min, y_min, x_max, y_max) and store it
                x_min = int(values[0])
                y_min = int(values[1])
                width = int(values[2])
                height = int(values[3])
                x_max = x_min + width
                y_max = y_min + height
                
                # Store class of annotation
                object_class = values[5]
                
                # Add segmentation data from the bounding box information 
                segmentation = [(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)]
                
                edited_line = f"{x_min},{y_min},{x_max},{y_max},{object_class},{segmentation}\n"
                file.write(edited_line)

    print("Files edited successfully!")


# UAV-SOD dateset ----------------------------------------------------------------------------------------

def xml_to_txt(input_folder, map_path="src/code_map.json"):
    """    
    Parameters:
      - input_folder (string): Path of the folder containing the XML files.
      - category_name_to_id (dict): Mapping from category names to IDs.
      - map_path (string): Path of the code_map json file that contains the codes for the classes

    """
    map = open(map_path)
    data = json.load(map)
    category_name_to_id = data['UAV_SOD_DRONE']['CATEGORY_ID_TO_NAME']
    map.close() 
    
    # Retrieve all XML files in the directory and initialize tqdm loop
    xml_files = [f for f in os.listdir(input_folder) if f.endswith('.xml')]
    for file in tqdm(xml_files, desc="Converting XML to TXT"):
        filename = os.path.join(input_folder, file)
        
        with open(filename, "r") as xml_file:
            xml_string = xml_file.read()
            python_dict = xmltodict.parse(xml_string)
            objects = python_dict['annotation']['object'] if isinstance(python_dict['annotation']['object'], list) else [python_dict['annotation']['object']]
            
            annotations = []
            for obj in objects:
                # Store bounding box data
                x_min = int(obj['bndbox']['xmin'])
                y_min = int(obj['bndbox']['ymin'])
                x_max = int(obj['bndbox']['xmax'])
                y_max = int(obj['bndbox']['ymax'])    
                
                # Store class annotation with code
                category_name = obj['name']
                category_code = next(int(key) for key, value in category_name_to_id.items() if value == category_name)
                
                # Add segmentation data from the bounding box information
                segmentation = [(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)]

                annotation = f"{x_min},{y_min},{x_max},{y_max},{category_code},{segmentation}"
                annotations.append(annotation)    
            
            annotations_text = "\n".join(annotations)   
            output_filename = os.path.splitext(file)[0] + ".txt"
            output_filepath = os.path.join(input_folder, output_filename)
            with open(output_filepath, "w") as txt_output:
                txt_output.write(annotations_text)
                
            os.remove(filename)



# City-Scapes dataset -----------------------------------------------------------------------------
def reorganize_cityscapes(city_scapes_images, city_scapes_annotations):
    """
    Parameters:
        - city_scapes_images (string): _description_
        - city_scapes_annotations (string): _description_
    """
    dest_dir = 'city-scapes-data'

    # Create the new directory structure
    for subset in ['train', 'test', 'val']:
        os.makedirs(os.path.join(dest_dir, subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, subset, 'annotations'), exist_ok=True)

        # Move and rename images
        initial_images = os.path.join(city_scapes_images, subset)
        destination_images = os.path.join(dest_dir, subset, 'images')
        
        for city_folder in os.listdir(initial_images):  
            full_city_folder_path = os.path.join(initial_images, city_folder)
            
            for image_file in os.listdir(full_city_folder_path):
                source_image_path      = os.path.join(full_city_folder_path, image_file)
                new_image_name         = image_file.replace('_leftImg8bit.png', '.png')
                destination_image_path = os.path.join(destination_images, new_image_name)
                # Move the image to new path with new name
                shutil.move(source_image_path, destination_image_path)


        # Move and rename annotations
        initial_annotations = os.path.join(city_scapes_annotations, subset)
        destination_annotations = os.path.join(dest_dir, subset, 'annotations')
        
        for city_folder in os.listdir(initial_annotations):
            full_city_folder_path = os.path.join(initial_annotations, city_folder)
            
            for annotation_file in os.listdir(full_city_folder_path):
                if annotation_file.endswith('_polygons.json'):
                    source_annotation_path      = os.path.join(full_city_folder_path, annotation_file)
                    new_annotations_name        = annotation_file.replace('_gtFine_polygons.json', '.json')
                    destination_annotation_path = os.path.join(destination_annotations, new_annotations_name)
                    # Move the annotation to new path with new name
                    shutil.move(source_annotation_path, destination_annotation_path)
                        
                        
def json_to_text(input_folder, output_text_path, map_path="src/code_map.json"):
    """
    Parameters:
        - input_json_path (string): Path of the folder containing the city folders with the json annotations.
        - output_text_path (string): Path of the folder containing the city folders with the images.
        - map_path (string): Path of the code_map json file that contains the codes for the classes
    """
    map = open(map_path)
    data = json.load(map)
    category_name_to_id = data['CITY_SCAPES']['CATEGORY_ID_TO_NAME']
    map.close()
    
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

    for json_file in tqdm(json_files, desc="Converting JSON to Text files"):
        input_json_path = os.path.join(input_folder, json_file)
        output_text_path = os.path.join(input_folder, json_file.replace('.json', '.txt'))
        
        with open(input_json_path, 'r') as file:
            data = json.load(file)

        with open(output_text_path, 'w') as file:
            for obj in data['objects']:
                label = obj['label']
                polygon = obj['polygon']
                
                # Calculate bounding box from polygon
                x_min = min(point[0] for point in polygon)
                x_max = max(point[0] for point in polygon)
                y_min = min(point[1] for point in polygon)
                y_max = max(point[1] for point in polygon)
                
                # Get class code using dictionary
                category_code = next(int(key) for key, value in category_name_to_id.items() if value == label)
                
                # Format polygon for output
                polygon_str = str([(x, y) for x, y in polygon])                
                file.write(f"{x_min},{y_min},{x_max},{y_max},{category_code},{polygon_str}\n")
        
        # Remove the JSON file after conversion
        os.remove(input_json_path)


def rename_files(base_folder):
    """
    Parameters:
        - base_folder (string): Path of the folder containing the city folders with the images.
    """
    image_folder = os.path.join(base_folder, 'image')
    annotations_folder = os.path.join(base_folder, 'annotations')
    
    # Get the list of image and text files
    image_files = [f for f in os.listdir(image_folder) if f.endswith('leftImg8bit.png')]
    annotation_files = [f for f in os.listdir(annotations_folder) if f.endswith('gtFine_polygons.txt')]
    
    # Rename image files with tqdm progress bar
    for img_file in tqdm(image_files, desc="Renaming image files"):
        identifier = img_file.split('_leftImg8bit.png')[0]
        new_img_name = f"{identifier}.png"
        os.rename(os.path.join(image_folder, img_file), os.path.join(image_folder, new_img_name))

    # Rename annotation files with tqdm progress bar
    for json_file in tqdm(annotation_files, desc="Renaming annotation files"):
        identifier = json_file.split('_gtFine_polygons.txt')[0]
        new_json_name = f"{identifier}.txt"
        os.rename(os.path.join(annotations_folder, json_file), os.path.join(annotations_folder, new_json_name))
    