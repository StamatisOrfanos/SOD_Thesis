
# import os
# import chardet
# import matplotlib.pyplot as plt

# def detect_encoding(file_path):
#     with open(file_path, 'rb') as file:
#         raw_data = file.read(5000)
#         result = chardet.detect(raw_data)
#         return result['encoding']

# def count_classes_in_annotations(file_path):
#     class_counts = {}
#     encoding = detect_encoding(file_path)
#     with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
#         for line in file:
#             try:
#                 parts = line.strip().split(',')
#                 class_code = int(parts[4])
#                 if class_code in class_counts:
#                     class_counts[class_code] += 1
#                 else:
#                     class_counts[class_code] = 1
#             except Exception as e:
#                 print(f"Error processing line in {file_path}: {e}")
#     return class_counts

# def process_dataset(base_path):
#     subsets = ['train', 'test', 'validation']
#     overall_counts = {}
#     if not os.path.exists(os.path.join(base_path, 'test')):
#         subsets.remove('test')
#     for subset in subsets:
#         subset_path = os.path.join(base_path, subset, 'annotations')
#         for annotation_file in os.listdir(subset_path):
#             file_path = os.path.join(subset_path, annotation_file)
#             class_counts = count_classes_in_annotations(file_path)
#             for class_code, count in class_counts.items():
#                 if class_code in overall_counts:
#                     overall_counts[class_code] += count
#                 else:
#                     overall_counts[class_code] = count
#     return overall_counts

# def plot_histogram(class_counts, dataset_name, class_mapping=None):
#     if class_mapping:
#         labels = {int(k): v for k, v in class_mapping.items()}
#         # Map counts to class names
#         named_counts = {labels[k]: v for k, v in class_counts.items() if k in labels}
#         keys = list(named_counts.keys())
#         values = list(named_counts.values())
#     else:
#         keys = list(class_counts.keys())
#         values = list(class_counts.values())

#     plt.figure(figsize=(10, 5))
#     plt.bar(keys, values, color='blue', width=0.5)
#     plt.xlabel('Class')
#     plt.ylabel('Number of Instances')
#     name = ""
    
#     if dataset_name == "coco2017":
#         name = "COCO 2017"
#     elif dataset_name == "uav_sod_data":
#         name = "UAV Small Object Detection"
#     else:
#         name = "VIS Drone"
    
#     plt.title(f'Class Distribution in {name}')
#     plt.tight_layout()    
#     plt.savefig(f"{dataset_name}_class_distribution.png")
#     plt.close()

# def main():
#     class_mappings = {
#         'uav_sod_data': {
#             "1": "building", "2": "vehicle", "3": "ship", "4": "pool", "5": "quarry",
#             "6": "well", "7": "house", "8": "cable-tower", "9": "mesh-cage", "10": "landslide"
#         },
#         'vis_drone_data': {
#             "0": "ignore", "1": "pedestrian", "2": "people", "3": "bicycle", "4": "car",
#             "5": "van", "6": "truck", "7": "tricycle", "8": "tricycle", "9": "bus", "10": "motor", "11": "others", "12": "obj"
#         }
#     }

#     datasets = {
#         'coco2017': 'data/coco2017',
#         'uav_sod_data': 'data/uav_sod_data',
#         'vis_drone_data': 'data/vis_drone_data'
#     }

#     for dataset_name, dataset_path in datasets.items():
#         print(f"Processing {dataset_name}...")
#         class_counts = process_dataset(dataset_path)
#         class_mapping = class_mappings.get(dataset_name)
#         plot_histogram(class_counts, dataset_name, class_mapping)
#         print(f"Completed {dataset_name}. Saved histogram as '{dataset_name}_class_distribution.png'.")

# if __name__ == "__main__":
#     main()


# import os
# path = "/Users/stamatiosorphanos/Documents/MCs_Thesis/SOD_Thesis/data/uav_sod_data/train/images"

# num_files = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
# print(num_files)