import numpy as np


class Anchors():
    def generate_anchors(feature_map_shapes, scales, aspect_ratios):
        """
        Parameters:
            - feature_map_shapes (list): List of different down-sampling levels
            - scales (list): List of range of scales that can cover various object sizes
            - aspect_ratios (list): Aspect ratios to cover different object shapes.
        """
        anchors = []
        
        for shape in feature_map_shapes:
            for scale in scales:
                for ratio in aspect_ratios:
                    # Compute anchor box dimensions
                    anchor_width = scale * np.sqrt(ratio)
                    anchor_height = scale / np.sqrt(ratio)
                    
                    for y in range(shape[0]):
                        for x in range(shape[1]):
                            cx = (x + 0.5) / shape[1]
                            cy = (y + 0.5) / shape[0]
                            
                            # Convert to (x_min, y_min, x_max, y_max)
                            x_min = cx - anchor_width / 2
                            y_min = cy - anchor_height / 2
                            x_max = cx + anchor_width / 2
                            y_max = cy + anchor_height / 2
                            anchors.append([x_min, y_min, x_max, y_max])

        return np.array(anchors)
