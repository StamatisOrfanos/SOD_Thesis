import os, shutil, json, xmltodict
import numpy as np
from tqdm import tqdm
from PIL import ImageOps, Image
import cv2, random
import numpy as np


# Pre-processing functions for all datasets ------------------------------------------------------------------------

def resize_data(base_path):
    """
    Parameters:
        base_path (string): Path of directory we want to resize the images and the annotations
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
        image_path (string): Path of the image we want to resize
        target_size (tuple, optional): Target size for the new resized image
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
        image_path (string): Path of the image we want to resize
        annotation_path (string): Path of the annotation file we want to update
        target_size (tuple, optional): Target size for the new resized image   
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
      images_path (str): The path to the directory containing the images.
      dataset_name (str): The name of the dataset for which the statistics are computed.
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
    data_type = ""
    
    if "train" in images_path:
        data_type = "train"
    elif "test" in images_path:
        data_type = "test"
    else:
        data_type = "validation"
            
    with open('src/' + '{}_{}_preprocessing.json'.format(dataset_name, data_type), 'w') as f:
        json.dump(stats, f, indent=4)



# COCO dataset ---------------------------------------------------------------------------------------------

def reorganize_coco_structure(base_dir):
    """
    Parameters:
        base_dir (string): Path of the COCO2017 base directory.
    """
    # Define original paths
    images_dir = os.path.join(base_dir, 'images')
    annotations_dir = os.path.join(base_dir, 'annotations')
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    # Create new directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    # Move train images and annotations
    shutil.move(os.path.join(images_dir, 'train_images'), os.path.join(train_dir, 'images'))
    for file_name in ['captions_train2017.json', 'instances_train2017.json', 'person_keypoints_train2017.json']:
        if "instances" in file_name:
            shutil.move(os.path.join(annotations_dir, file_name), train_dir)
        else:
            shutil.delete(file_name)

    # Move validation images and annotations
    shutil.move(os.path.join(images_dir, 'validation_images'), os.path.join(validation_dir, 'images'))
    for file_name in ['captions_val2017.json', 'instances_val2017.json', 'person_keypoints_validation2017.json']:
        if "instances" in file_name:
            shutil.move(os.path.join(annotations_dir, file_name), validation_dir)
        else:
            shutil.delete(file_name)

    os.rmdir(images_dir)
    os.rmdir(annotations_dir)


def convert_coco_annotations(input_json, output_dir):
    """
    Parameter:
        input_json (string): Path of the directory containing the original annotation 
        output_dir (string): Path of the directory containing the new text annotations
    """
    # Load the original COCO annotations
    with open(input_json, "r") as f:
        data = json.load(f)

    # Prepare output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each annotation associated with images
    annotations_by_image = {ann["image_id"]: [] for ann in data["annotations"]}
    for annotation in data["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)
        
    
    # Write annotations in separate files for each image
    for image_id, annotations in annotations_by_image.items():
        image_file_name = f"{image_id}.txt"
        with open(os.path.join(output_dir, image_file_name), "w") as file:
            for annotation in annotations:
                # Extract bounding box and convert to integer x_min, y_min, x_max, y_max
                bbox = annotation["bbox"]
                x_min, y_min, width, height =  map(int, map(round, bbox))
                x_max, y_max = x_min + width, y_min + height

                # Category ID
                category_id = annotation['category_id']            
                
                # Segmentation data
                if "segmentation" in annotation and annotation["segmentation"]: 
                    segmentation = annotation["segmentation"][0]
                    integer_segmentation = [int(round(x)) for x in segmentation]
                    pairs = list(zip(integer_segmentation[::2], integer_segmentation[1::2]))
                else:
                    pairs = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

                formatted_pairs = str(pairs).replace(' ', '')

                # Write to file
                file.write(f"{x_min},{y_min},{x_max},{y_max},{category_id},{formatted_pairs}\n")
                

def remove_leading_zeros(input_folder):
    """
    Parameters:
        input_folder (string): Path of the folder containing the images
    """
    files = os.listdir(input_folder)
    
    for filename in tqdm(files, desc="Renaming images - Removing zeros prefix"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            new_filename = filename.lstrip('0')
            if new_filename != filename:
                os.rename(filename, new_filename)
    

# Vis-Drone dateset ----------------------------------------------------------------------------------------

def extract_annotation_values(input_folder):
    """    
    Parameters:
      input_folder (str): Path of the folder containing the text files.
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
      input_folder (str): Path of the folder containing the XML files.
      category_name_to_id (dict): Mapping from category names to IDs.
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
