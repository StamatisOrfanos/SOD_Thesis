import os, json, xmltodict
import numpy as np
from tqdm import tqdm
from PIL import ImageOps, Image


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
            for line in lines:
                values = line.strip().split(',')                
                # Convert (x_min, y_min, width, height) to (x_min, y_min, x_max, y_max)
                x_min = int(values[0])
                y_min = int(values[1])
                width = int(values[2])
                height = int(values[3])
                x_max = x_min + width
                y_max = y_min + height
                object_class = values[5]
                edited_line = f"{x_min},{y_min},{x_max},{y_max},{object_class}\n"
                file.write(edited_line)

    print("Files edited successfully!")


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
                xmin = int(obj['bndbox']['xmin'])
                ymin = int(obj['bndbox']['ymin'])
                xmax = int(obj['bndbox']['xmax'])
                ymax = int(obj['bndbox']['ymax'])        
                category_name = obj['name']
                category_code = next(int(key) for key, value in category_name_to_id.items() if value == category_name)
                annotation = f"{xmin},{ymin},{xmax},{ymax},{category_code}"
                annotations.append(annotation)    
            
            annotations_text = "\n".join(annotations)   
            output_filename = os.path.splitext(file)[0] + ".txt"
            output_filepath = os.path.join(input_folder, output_filename)
            with open(output_filepath, "w") as txt_output:
                txt_output.write(annotations_text)
                
            # Optionally, delete the XML file
            os.remove(filename)


def resize_images(base_dir, source_dir, target_dir, target_size=(600, 600)):
    """
    Parameters:
      base_dir (str): The path to the base directory containing all the data.
      source_dir (str): The path to the directory containing the original images.
      target_dir (str): The path to the directory where resized images will be saved.
      target_size (tuple): The target size (width, height) for the resized images.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                img_path = os.path.join(source_dir, filename)
                image = Image.open(img_path)
                original_width, original_height = image.size

                # Calculate the scaling factor and resize
                scale = min(target_size[0] / original_width, target_size[1] / original_height)
                new_size = (int(original_width * scale), int(original_height * scale))
                image = image.resize(new_size, Image.ANTIALIAS)

                # Determine padding
                padding_width = (target_size[0] - new_size[0]) // 2
                padding_height = (target_size[1] - new_size[1]) // 2
                padding = (padding_width, padding_height, target_size[0] - new_size[0] - padding_width, target_size[1] - new_size[1] - padding_height)

                # Find most common color for padding
                colors = image.convert('RGB').getcolors(maxcolors=new_size[0]*new_size[1])
                most_common_color = max(colors, key=lambda item: item[0])[1] if colors else (255, 255, 255)
                image_with_padding = ImageOps.expand(image, border=padding, fill=most_common_color)
                image_with_padding.save(os.path.join(target_dir, filename))
                
                # SOS ----------------------------------------------------------------------------------------------
                # SOS: Since we are going to resize the images we have to remap the bounding box coordinates as well
                # SOS ----------------------------------------------------------------------------------------------
                resize_bounding_boxes(base_dir, filename, original_width, original_height) 
  
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print("Successfully resized images and updated annotations")
    



def resize_bounding_boxes(base_dir, filename, original_width, original_height, target_size=(600,600)):
    """    
    Parameters:
      base_dir (str): The path to the base directory containing all the data.
      original_width (int): The original width of the image.
      original_height (int): The original height of the image.
      target_size (tuple): The target size (width, height) for the resized image.
    """
    annotation_dir = os.path.join(base_dir, 'annotations')
    
    annotation_filename = os.path.splitext(filename)[0] + '.txt'
    ann_path = os.path.join(annotation_dir, annotation_filename)
    
    if os.path.exists(ann_path):
        with open(ann_path, 'r+') as ann_file:
            lines = ann_file.readlines()
            ann_file.seek(0)
            ann_file.truncate()
            for line in lines:
                x_min, y_min, x_max, y_max, obj_class = map(float, line.split(','))
                x_min_new = (x_min / original_width) * target_size[0]
                x_max_new = (x_max / original_width) * target_size[0]
                y_min_new = (y_min / original_height) * target_size[1]
                y_max_new = (y_max / original_height) * target_size[1]
                new_line = f"{x_min_new},{y_min_new},{x_max_new},{y_max_new},{int(obj_class)}\n"
                ann_file.write(new_line)


def compute_mean_std(images_path, dataset_name):
    """    
    Parameters:
      images_path (str): The path to the directory containing the images.
      dataset_name (str): The name of the dataset for which the statistics are computed.
    """
    pixel_data = []
    
    for filename in os.listdir(images_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
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
