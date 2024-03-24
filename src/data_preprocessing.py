import sys, os, json, shutil, xmltodict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from tqdm import tqdm
import ImageOps
from PIL import Image
from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from sahi.utils.file import save_json



    
def annotation_processing(base_folder, output_file_path, dataset, coco_mapper_path='src/coco_config.json'):
    """
    Parameters:
       - data_folder_dir (str): base VisDrone folder path
       - output_file_path (str): Output file path
    """
    # Load the coco mapper from the 
    map = open(coco_mapper_path)
    data = json.load(map)
    
    
    if dataset == "VIS_DRONE":
        CATEGORY_ID_TO_NAME   =  data['VIS_DRONE']['CATEGORY_ID_TO_NAME']
        CATEGORY_ID_REMAPPING =  data['VIS_DRONE']['CATEGORY_ID_REMAPPING']
        NAME_TO_COCO_CATEGORY =  data['VIS_DRONE']['NAME_TO_COCO_CATEGORY']
    elif dataset == "UAV_SOD_DRONE":
        CATEGORY_ID_TO_NAME   =  data['UAV_SOD_DRONE']['CATEGORY_ID_TO_NAME']
        CATEGORY_ID_REMAPPING =  data['UAV_SOD_DRONE']['CATEGORY_ID_REMAPPING']
        NAME_TO_COCO_CATEGORY =  data['UAV_SOD_DRONE']['NAME_TO_COCO_CATEGORY']
    else:
        CATEGORY_ID_TO_NAME   =  None
        CATEGORY_ID_REMAPPING =  None
        NAME_TO_COCO_CATEGORY =  None
    
    map.close()  
    
    # Get the path for the images and annotations folders
    image_path          = str(Path(base_folder) / "images")
    annotations_path    = str(Path(base_folder) / "annotations")
    images_list = os.listdir(image_path)
    Path(output_file_path).parents[0].mkdir(parents=True, exist_ok=True)
    
    # Get the annotation mapping from the VisDrone to COCO dataset
    category_id_remapping = CATEGORY_ID_REMAPPING
    
    # Create the COCO object and append the categories
    coco = Coco()
    for category_id, category_name in CATEGORY_ID_TO_NAME.items():
        if category_id in category_id_remapping.keys():
            remapped_category_id = category_id_remapping[category_id]
            coco_category = NAME_TO_COCO_CATEGORY[category_name]
            coco.add_category(
                CocoCategory(
                    id=int(remapped_category_id),
                    name=coco_category["name"],
                    supercategory=coco_category["supercategory"],
                )
            )

    # Convert VisDrone annotations to COCO equivalents 
    for image_filename in tqdm(images_list):
        
        # For each image get the annotations
        current_image_path = str(Path(image_path) / image_filename)
        annotation_name = image_filename.split(".jpg")[0] + ".txt"
        annotation_path = str(Path(annotations_path) / annotation_name)
        
        # Create the COCO image file that we are going to map in the annotations 
        image = Image.open(current_image_path)
        coco_image_name = str(Path(image_path)).split(str(Path(base_folder)))[1]
        
        if coco_image_name[0] == os.sep: coco_image_name = coco_image_name[1:]            
        coco_image = CocoImage(file_name=coco_image_name, height=image.size[1], width=image.size[0])
        
        # Parse annotation file and extract the important information like the bounding box and COCO category 
        file = open(annotation_path, "r")
        lines = file.readlines()
        
        for line in lines:
            # Parse bounding box [x_min, y_min, width, height]
            new_line = line.strip("\n").split(",")
            bbox = [int(new_line[0]), int(new_line[1]), int(new_line[2]), int(new_line[3])]
                  
            # Parse category Id and Name
            category_id = new_line[5]
            if category_id in category_id_remapping.keys():
                category_name = CATEGORY_ID_TO_NAME[category_id]
                remapped_category_id = category_id_remapping[category_id]
            else:
                continue
            
            coco_annotation = CocoAnnotation.from_coco_bbox(bbox=bbox, category_id=int(remapped_category_id), category_name=category_name,)                
            if coco_annotation.area > 0: coco_image.add_annotation(coco_annotation)

        coco.add_image(coco_image)
    save_path = str(Path(base_folder) / output_file_path)
    save_json(data=coco.json, save_path=save_path)




def xml_to_txt(input_folder, coco_mapper_path='src/coco_config.json', dataset="UAV_SOD_DRONE"):
    """    
    Parameters:
    - input_folder (str): Path of the folder containing the XML files.
    - category_name_to_id (dict): Mapping from category names to IDs.
    """
    # Load the coco mapper from the 
    map = open(coco_mapper_path)
    data = json.load(map)
       
    if dataset == "VIS_DRONE":
        CATEGORY_ID_TO_NAME = data['VIS_DRONE']['CATEGORY_ID_TO_NAME']
    else:
        CATEGORY_ID_TO_NAME = data['UAV_SOD_DRONE']['CATEGORY_ID_TO_NAME']
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
                category_id = list(filter(lambda x: CATEGORY_ID_TO_NAME[x] == category_name, CATEGORY_ID_TO_NAME))[0]  
                
                
                annotation = f"{xmin},{ymin},{width},{height},0,{category_id}"
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
    - source_dir (str): The path to the directory containing the original images.
    - target_dir (str): The path to the directory where resized images will be saved.
    - target_size (tuple): The target size (width, height) for the resized images.
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
                    left_margin = (target_size[0] - image.width) / 2
                    top_margin = (target_size[1] - image.height) / 2
                    right_margin = (target_size[0] - image.width) - left_margin
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
    - images_path (str): The path to the directory containing the images.
    - dataset_name (str): The name of the dataset for which the statistics are computed.
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
    with open('preprocessing.json', 'w') as f:
        json.dump(stats, f, indent=4)
