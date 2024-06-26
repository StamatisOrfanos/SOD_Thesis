import numpy as np

class Anchors:
    @staticmethod
    def generate_anchors(feature_map_shapes, scales, aspect_ratios, image_size=300):
        anchors = []
        for shape in feature_map_shapes:
            fm_height, fm_width = shape
            stride_height = image_size / fm_height
            stride_width = image_size / fm_width

            for y in range(fm_height):
                for x in range(fm_width):
                    cy = (y + 0.5) * stride_height
                    cx = (x + 0.5) * stride_width

                    for scale in scales:
                        for ratio in aspect_ratios:
                            anchor_height = scale * np.sqrt(ratio)
                            anchor_width = scale / np.sqrt(ratio)

                            x_min = cx - anchor_width / 2
                            y_min = cy - anchor_height / 2
                            x_max = cx + anchor_width / 2
                            y_max = cy + anchor_height / 2

                            # Ensure anchors are within image bounds
                            x_min = max(0, x_min)
                            y_min = max(0, y_min)
                            x_max = min(image_size, x_max)
                            y_max = min(image_size, y_max)

                            anchors.append([x_min, y_min, x_max, y_max])

        return np.array(anchors)