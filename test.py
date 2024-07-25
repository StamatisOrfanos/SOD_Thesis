# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision.models as models
# from efficientnet_pytorch import EfficientNet


# class EFPN(nn.Module):
#     def __init__(self, in_channels, hidden_dim, num_boxes, num_classes):
#         super(EFPN, self).__init__()
        
#         self.backbone = EfficientNet.from_pretrained('efficientnet-b7')
#         self.ftt_model = FTT()

#         # Define FPN convolution layers to match channel dimensions if necessary
#         self.conv_c2_prime = nn.Conv2d(32, 256, kernel_size=1)  
#         self.conv_c2       = nn.Conv2d(48, 256, kernel_size=1)  
#         self.conv_c3       = nn.Conv2d(80, 256, kernel_size=1)  
#         self.conv_c4       = nn.Conv2d(224, 256, kernel_size=1) 
#         self.conv_c5       = nn.Conv2d(640, 256, kernel_size=1) 

#         # Define FPN lateral layers
#         self.lateral_p5       = nn.Conv2d(640, 256, kernel_size=1)
#         self.lateral_p4       = nn.Conv2d(224, 256, kernel_size=1)
#         self.lateral_p3       = nn.Conv2d(80, 256, kernel_size=1) 
#         self.lateral_p2       = nn.Conv2d(48, 256, kernel_size=1) 
#         self.lateral_p2_prime = nn.Conv2d(32, 256, kernel_size=1) 

#         # Define FPN top-down pathway
#         self.top_down_p5 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.top_down_p4 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.top_down_p3 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.top_down_p2 = nn.Upsample(scale_factor=2, mode='nearest')

#         self.bounding_box = BoundingBoxGenerator(in_channels, num_classes)


#     def forward(self, image):
#         c2_prime, c2, c3, c4, c5 = self.backbone_features(image)

#         # Process feature maps through FPN
#         p5 = self.lateral_p5(c5)
#         upsampled_p5 = self.top_down_p5(p5)
        
#         p4 = self.lateral_p4(c4) + upsampled_p5
#         upsampled_p4 = F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        
#         p3 = self.lateral_p3(c3) + upsampled_p4
#         upsampled_p3 = F.interpolate(p3, size=c2.shape[2:], mode='nearest')
        
#         p2 = self.lateral_p2(c2) + upsampled_p3

#         # FTT operations for P3' and P2' here
#         p3_prime = self.ftt_model(p2, p3)

#         # Process c2_prime through its convolution layer
#         c2_prime_processed = self.conv_c2_prime(c2_prime)
#         upsampled_p3_prime = self.top_down_p2(p3_prime)
#         p2_prime = upsampled_p3_prime + c2_prime_processed
            
#         # Create the mask for the spatially richest feature map p2_prime
#         feature_maps = [p2_prime, p2, p3, p4, p5]
#         bounding_box_regressions, class_scores = self.bounding_box(feature_maps)
        
#         # Return the feature map pyramid and the mask
#         return feature_maps, bounding_box_regressions, class_scores


#     def backbone_features(self, image):
#         # Get feature maps from the EfficientNet backbone
#         endpoints = self.backbone.extract_endpoints(image)
#         c2_prime  = endpoints['reduction_1']
#         c2        = endpoints['reduction_2']
#         c3        = endpoints['reduction_3']
#         c4        = endpoints['reduction_4']
#         c5        = endpoints['reduction_5']
#         return c2_prime, c2, c3, c4, c5
    
    
# class FTT(nn.Module):
#     def __init__(self):
#         super(FTT, self).__init__()
#         self.content_extractor = ContentExtractor(256, 256, num_layers=3)
#         self.texture_extractor = TextureExtractor(256, 256, num_layers=3)
#         self.subpixel_conv     = SubPixelConv(256, 256, upscale_factor=2)
    
#     def forward(self, p2, p3):
#         content_features = self.content_extractor(p3)
#         upsampled_content = self.subpixel_conv(content_features)
#         texture_features = self.texture_extractor(p2)
#         combined_features = upsampled_content + texture_features
#         return combined_features


# class ContentExtractor(nn.Module):
#     def __init__(self, in_channels, out_channels, num_layers):
#         super().__init__()
#         layers = []
#         for _ in range(num_layers):
#             layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
#             layers.append(nn.BatchNorm2d(out_channels))
#             layers.append(nn.ReLU(inplace=True))
#             in_channels = out_channels
#         self.layers = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.layers(x)


# class TextureExtractor(nn.Module):
#     def __init__(self, in_channels, out_channels, num_layers):
#         super().__init__()
#         layers = []
#         for _ in range(num_layers):
#             layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
#             layers.append(nn.BatchNorm2d(out_channels))
#             layers.append(nn.ReLU(inplace=True))
#             in_channels = out_channels
#         self.layers = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.layers(x)


# class SubPixelConv(nn.Module):
#     def __init__(self, in_channels, out_channels, upscale_factor):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
#         self.upscale_factor = upscale_factor
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = F.pixel_shuffle(x, self.upscale_factor)
#         return x


# class BoundingBoxGenerator(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(BoundingBoxGenerator, self).__init__()
#         self.num_classes = num_classes
#         self.classification_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)
#         self.regression_head = nn.Conv2d(in_channels, 4, kernel_size=1)

#     def forward(self, feature_maps):
#         class_scores = [self.classification_head(fm) for fm in feature_maps]
#         bounding_boxes = [self.regression_head(fm) for fm in feature_maps]        
#         class_scores = torch.cat([cs.view(cs.size(0), -1, self.num_classes) for cs in class_scores], dim=1)
#         bounding_boxes = torch.cat([bb.view(bb.size(0), -1, 4) for bb in bounding_boxes], dim=1)
        
#         return bounding_boxes, class_scores


# def generate_anchors(feature_map_shapes, scales, aspect_ratios):
#     anchors = []

#     for shape in feature_map_shapes:
#         for scale in scales:
#             for ratio in aspect_ratios:
#                 # Compute anchor box dimensions
#                 anchor_width = scale * np.sqrt(ratio)
#                 anchor_height = scale / np.sqrt(ratio)

#                 for y in range(shape[0]):
#                     for x in range(shape[1]):
#                         cx = (x + 0.5) / shape[1]
#                         cy = (y + 0.5) / shape[0]

#                         # Convert to (x_min, y_min, x_max, y_max)
#                         x_min = cx - anchor_width / 2
#                         y_min = cy - anchor_height / 2
#                         x_max = cx + anchor_width / 2
#                         y_max = cy + anchor_height / 2

#                         anchors.append([x_min, y_min, x_max, y_max])

#     return np.array(anchors)



# def match_anchors_to_ground_truth(anchors, gt_boxes, gt_labels, num_classes):
#     num_anchors = anchors.shape[0]
#     num_gt_boxes = gt_boxes.shape[0]

#     # Initialize matched labels and boxes
#     matched_labels = torch.zeros((num_anchors,), dtype=torch.long)
#     matched_boxes = torch.zeros((num_anchors, 4))

#     # Compute IoUs
#     ious = torch.zeros((num_anchors, num_gt_boxes))
#     for i, anchor in enumerate(anchors):
#         ious[i, :] = calculate_iou(anchor, gt_boxes)

#     # Find the best match for each anchor
#     max_ious, max_indices = ious.max(dim=1)

#     for i in range(num_anchors):
#         if max_ious[i] > 0.5:  # IoU threshold
#             matched_labels[i] = gt_labels[max_indices[i]]
#             matched_boxes[i, :] = gt_boxes[max_indices[i], :]
#         else:
#             matched_labels[i] = num_classes - 1  # Background class

#     return matched_boxes, matched_labels


# def calculate_iou(anchor, gt_boxes):
#     # Anchor box corners
#     anchor_x1, anchor_y1, anchor_x2, anchor_y2 = anchor

#     # Ground truth box corners
#     gt_x1 = gt_boxes[:, 0]
#     gt_y1 = gt_boxes[:, 1]
#     gt_x2 = gt_boxes[:, 2]
#     gt_y2 = gt_boxes[:, 3]

#     inter_x1 = torch.max(anchor_x1, gt_x1)
#     inter_y1 = torch.max(anchor_y1, gt_y1)
#     inter_x2 = torch.min(anchor_x2, gt_x2)
#     inter_y2 = torch.min(anchor_y2, gt_y2)

#     inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
#     anchor_area = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
#     gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)

#     union_area = anchor_area + gt_area - inter_area
#     iou = inter_area / union_area
#     return iou


# class ObjectDetectionLoss(nn.Module):
#     def __init__(self):
#         super(ObjectDetectionLoss, self).__init__()
#         self.classification_loss = nn.CrossEntropyLoss()
#         self.regression_loss = nn.SmoothL1Loss()

#     def forward(self, pred_bboxes, pred_scores, gt_bboxes, gt_labels):
#         cls_loss = self.classification_loss(pred_scores, gt_labels)        
#         reg_loss = self.regression_loss(pred_bboxes, gt_bboxes)
#         total_loss = cls_loss + reg_loss
#         return total_loss



# ---------------------------------------------------------------------------------------------------------------------------------------

# Example usage
# feature_map_shapes = [(75, 75), (38, 38), (19, 19)]
# scales = [32, 64, 128]
# aspect_ratios = [1, 2]
# anchors = generate_anchors(feature_map_shapes, scales, aspect_ratios)
# anchors = torch.tensor(anchors, dtype=torch.float32)

# # Create dummy training data
# batch_size = 2
# num_classes = 21  # 20 classes + 1 background class
# num_boxes = 10  # Number of boxes per image

# images = torch.rand(batch_size, 3, 600, 600)

# # Generate dummy annotations
# gt_bboxes = torch.rand(batch_size, num_boxes, 4) * 600  # Scale boxes to image size
# gt_labels = torch.randint(0, 20, (batch_size, num_boxes))

# model = EFPN(in_channels=256, hidden_dim=256, num_boxes=9, num_classes=num_classes)
# loss_fn = ObjectDetectionLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)


# model.train()
# for epoch in range(1):
#     optimizer.zero_grad()
#     feature_maps, pred_bboxes, pred_scores = model(images)
    
#     print(pred_bboxes.size())
#     print(pred_scores.size())
    
    
    # batch_loss = 0
    # for i in range(batch_size):
    #     gt_boxes = gt_bboxes[i]
    #     gt_lbls = gt_labels[i]

    #     matched_boxes, matched_labels = match_anchors_to_gt(anchors, gt_boxes, gt_lbls, num_classes)

    #     # Flatten the predictions and labels to match dimensions
    #     pred_scores_flat = pred_scores.view(-1, num_classes)
    #     matched_labels_flat = matched_labels.view(-1)

    #     pred_bboxes_flat = pred_bboxes.view(-1, 4)
    #     matched_boxes_flat = matched_boxes.view(-1, 4)

    #     # Calculate the loss for this image
    #     loss = loss_fn(pred_bboxes_flat, pred_scores_flat, matched_boxes_flat, matched_labels_flat)
    #     batch_loss += loss

    # batch_loss /= batch_size
    # batch_loss.backward()
    # optimizer.step()
    # print(f'Epoch {epoch+1}, Loss: {batch_loss.item()}')


import os
import chardet
import matplotlib.pyplot as plt

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read(5000)
        result = chardet.detect(raw_data)
        return result['encoding']

def count_classes_in_annotations(file_path):
    class_counts = {}
    encoding = detect_encoding(file_path)
    with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
        for line in file:
            try:
                parts = line.strip().split(',')
                class_code = int(parts[4])
                if class_code in class_counts:
                    class_counts[class_code] += 1
                else:
                    class_counts[class_code] = 1
            except Exception as e:
                print(f"Error processing line in {file_path}: {e}")
    return class_counts

def process_dataset(base_path):
    subsets = ['train', 'test', 'validation']
    overall_counts = {}
    if not os.path.exists(os.path.join(base_path, 'test')):
        subsets.remove('test')
    for subset in subsets:
        subset_path = os.path.join(base_path, subset, 'annotations')
        for annotation_file in os.listdir(subset_path):
            file_path = os.path.join(subset_path, annotation_file)
            class_counts = count_classes_in_annotations(file_path)
            for class_code, count in class_counts.items():
                if class_code in overall_counts:
                    overall_counts[class_code] += count
                else:
                    overall_counts[class_code] = count
    return overall_counts

def plot_histogram(class_counts, dataset_name, class_mapping=None):
    if class_mapping:
        labels = {int(k): v for k, v in class_mapping.items()}
        # Map counts to class names
        named_counts = {labels[k]: v for k, v in class_counts.items() if k in labels}
        keys = list(named_counts.keys())
        values = list(named_counts.values())
    else:
        keys = list(class_counts.keys())
        values = list(class_counts.values())

    plt.figure(figsize=(10, 5))
    plt.bar(keys, values, color='blue', width=0.5)
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    name = ""
    
    if dataset_name == "coco2017":
        name = "COCO 2017"
    elif dataset_name == "uav_sod_data":
        name = "UAV Small Object Detection"
    else:
        name = "VIS Drone"
    
    plt.title(f'Class Distribution in {name}')
    plt.tight_layout()    
    plt.savefig(f"{dataset_name}_class_distribution.png")
    plt.close()

def main():
    class_mappings = {
        'uav_sod_data': {
            "1": "building", "2": "vehicle", "3": "ship", "4": "pool", "5": "quarry",
            "6": "well", "7": "house", "8": "cable-tower", "9": "mesh-cage", "10": "landslide"
        },
        'vis_drone_data': {
            "0": "ignore", "1": "pedestrian", "2": "people", "3": "bicycle", "4": "car",
            "5": "van", "6": "truck", "7": "tricycle", "8": "tricycle", "9": "bus", "10": "motor", "11": "others", "12": "obj"
        }
    }

    datasets = {
        'coco2017': 'data/coco2017',
        'uav_sod_data': 'data/uav_sod_data',
        'vis_drone_data': 'data/vis_drone_data'
    }

    for dataset_name, dataset_path in datasets.items():
        print(f"Processing {dataset_name}...")
        class_counts = process_dataset(dataset_path)
        class_mapping = class_mappings.get(dataset_name)
        plot_histogram(class_counts, dataset_name, class_mapping)
        print(f"Completed {dataset_name}. Saved histogram as '{dataset_name}_class_distribution.png'.")

if __name__ == "__main__":
    main()

