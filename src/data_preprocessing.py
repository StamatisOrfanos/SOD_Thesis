import os
import shutil
from PIL import Image
from concurrent.futures import ThreadPoolExecutor



def resize_image(image_path, output_folder, size=(600, 600)):
    with Image.open(image_path) as image:
        image = image.resize(size)
        image.save(os.path.join(output_folder, os.path.basename(image_path)))



def process_images(input_folder, output_folder, size=(600, 600)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with ThreadPoolExecutor() as executor:
        for image_name in os.listdir(input_folder):
            image_path = os.path.join(input_folder, image_name)
            executor.submit(resize_image, image_path, output_folder, size)

    shutil.rmtree(input_folder)
