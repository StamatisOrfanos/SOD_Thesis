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
                edited_line = ','.join(values[:5]) + '\n'                
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
                width = xmax - xmin
                height = ymax - ymin                
                category_name = obj['name']
                category_code = next(int(key) for key, value in category_name_to_id.items() if value == category_name)
                annotation = f"{xmin},{ymin},{width},{height},{category_code}"
                annotations.append(annotation)
            
            
            annotations_text = "\n".join(annotations)   
            output_filename = os.path.splitext(file)[0] + ".txt"
            output_filepath = os.path.join(input_folder, output_filename)
            with open(output_filepath, "w") as txt_output:
                txt_output.write(annotations_text)
                
            # Optionally, delete the XML file
            os.remove(filename)


def resize_images(source_dir, target_dir, target_size=(600, 600)):
    """
    Parameters:
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
                width, height = image.size
                
                if width > 600 and height > 600:       
                    # In case the image is bigger than expected resize, while maintaining aspect ratio and save it to the target directory
                    image.thumbnail(target_size, Image.ANTIALIAS)                    
                    image.save(os.path.join(target_dir, filename))
                else:
                    # In case the image is smaller than expected increase with a "informed" padding based on the most common color to create better context
                    image.thumbnail(target_size, Image.ANTIALIAS)
                    result = image.convert('RGB').getcolors(image.width * image.height)
                    most_common_color = max(result, key=lambda item: item[0])[1]

                    # Calculate accurate padding that needs to be added
                    left_margin   = (target_size[0] - image.width) / 2
                    top_margin    = (target_size[1] - image.height) / 2
                    right_margin  = (target_size[0] - image.width) - left_margin
                    bottom_margin = (target_size[1] - image.height) - top_margin

                    # Add padding and save it to the target directory 
                    img_with_padding = ImageOps.expand(image, border=(int(left_margin), int(top_margin), int(right_margin), int(bottom_margin)), 
                                                       fill=most_common_color)
                    img_with_padding.save(os.path.join(target_dir, filename))
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")


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
    with open('{}_preprocessing.json'.format(dataset_name), 'w') as f:
        json.dump(stats, f, indent=4)
