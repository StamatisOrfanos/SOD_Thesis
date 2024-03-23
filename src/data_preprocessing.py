import sys, os, json, shutil, xmltodict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

import fire
from PIL import Image
from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from sahi.utils.file import save_json



    
def annotation_processing(base_folder, output_file_path, dataset, coco_mapper_path='src/coco_config.json'):
    """
    Parameters:
        data_folder_dir (str): base VisDrone folder path
        output_file_path (str): Output file path
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

        

                
           
        
#  ------------- Keep for future reference ------------- #
# base_dir = 'data/vis_drone_data'
# train_folder = str(Path(base_dir) / "train")
# validation_folder = str(Path(base_dir) / "validation")
# # Generate COCO JSON files
# annotation_processing(validation_folder, "val_annotations.json")
#  ---------------------------------------------------- #
# xml_to_txt("uav_sod_data/test/annotations")
# annotation_processing("uav_sod_data/test", "test_annotations.json", "UAV_SOD_DRONE")
