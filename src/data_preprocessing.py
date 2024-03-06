import sys, os, json, shutil, xmltodict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pycocotools.coco import COCO
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader




class CocoDataset(Dataset):
    def __init__(self, root, annotation, transform=None):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        # Note: You'll need to convert COCO's annotations to your target mask format here

        if self.transform is not applied:
            img = self.transform(img)

        # Example: Convert annotations to masks and apply preprocessing
        # masks, categories = convert_coco_poly_to_mask(coco_annotation, img.size)

        return img, masks, categories

    def __len__(self):
        return len(self.ids)




def data_pre_processing(data_folder):
    """
    Parameters:
        data_folder (string): path of the data folder that includes all the datasets
    """
    for subfolder_name in os.listdir(data_folder):
        
        subfolder_path = os.path.join(data_folder, subfolder_name)
        
        # Check if the subfolder is a directory
        if os.path.isdir(subfolder_path):
            
            # Check if the subfolder contains both train, test and validation folders
            if all(os.path.isdir(os.path.join(subfolder_path, folder)) for folder in ['train', 'validation'] or ['train', 'test', 'validation']):
                
                # Iterate through train, test and validation folders
                for folder_type in ['train', 'test', 'validation']:
                    if folder_type not in os.listdir(subfolder_path): continue                 
                    
                    # Image Processing - Resize image and save
                    image_path = os.path.join(subfolder_path, folder_type, 'images')
                    print(f"Dataset {subfolder_name}:  {folder_type}/images: {len(os.listdir(image_path))} files")
                    process_images(image_path)                 
                    
                    
                    # Annotation Processing - XML to JSON transformation
                    annotations_path = os.path.join(subfolder_path, folder_type, 'annotations')
                    print(f"Dataset {subfolder_name}:  {folder_type}/annotations: {len(os.listdir(annotations_path))} files")
                    xml_to_json(annotations_path)


def resize_image(input_folder, size=(600, 600)):
    """
    Parameters:
        input_folder (string): path of the folder containing the image files
        size (tuple, optional): size of the images we are going to resize (we use efficientNet-b7 pre-trained model)
    """
    with Image.open(input_folder) as image:
        image = image.resize(size, Image.NEAREST)
        image.save(os.path.join(input_folder, os.path.basename(input_folder)))



def process_images(input_folder, size=(600, 600)):
    """
    Parameters:
        input_folder (string): path of the folder containing the image files
        size (tuple, optional): size of the images we are going to resize (we use efficientNet-b7 pre-trained model)
    """
    with ThreadPoolExecutor() as executor:
        for image_name in os.listdir(input_folder):
            image_path = os.path.join(input_folder, image_name)
            executor.submit(resize_image, image_path, image_path, size)

    shutil.rmtree(input_folder)


def xml_to_json(input_folder):
    """
    Parameters:
        input_folder (string): path of the folder containing the xml files
    """
        
    for file in os.listdir(input_folder):
        filename = os.path.join(input_folder, file)
        
        if filename.endswith(".xml"):
            with open(filename, "r") as xml_file:
                xml_string = xml_file.read()
                python_dict = xmltodict.parse(xml_string)
                objects = python_dict['annotation']['object']
                annotations = []
                for obj in objects:
                    annotation = {
                        'name': obj['name'],
                        'bbox': {
                            'xmin': int(obj['bndbox']['xmin']),
                            'ymin': int(obj['bndbox']['ymin']),
                            'xmax': int(obj['bndbox']['xmax']),
                            'ymax': int(obj['bndbox']['ymax'])
                        }
                    }
                    annotations.append(annotation)
                json_data = json.dumps(annotations)
                # Save JSON file
                output_filename = os.path.splitext(file)[0] + ".json"
                output_filepath = os.path.join(input_folder, output_filename)
                with open(output_filepath, "w") as json_output:
                    json_output.write(json_data)
                # Delete XML file
                os.remove(filename)
        else:
            continue