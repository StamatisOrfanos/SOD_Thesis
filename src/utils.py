from sklearn.metrics import precision_recall_curve, auc

def average_precision(y_true, y_scores):
    """Calculate the average precision for a class."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    # Calculate area under the curve to get average precision
    return auc(recall, precision)

def mean_average_precision(y_trues, y_scores):
    """Calculate the mean average precision across all classes and return mean average precision across all classes.
    
    Parameters:
        y_trues: A list of arrays, where each array contains the true binary labels for a class.
        y_scores: A list of arrays, where each array contains the predicted scores for a class.
    """
    aps = []
    for y_true, y_score in zip(y_trues, y_scores):
        aps.append(average_precision(y_true, y_score))
    return sum(aps) / len(aps)


def intersection_over_union(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
	yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
	boxBArea = (boxB[2] + 1) * (boxB[3] + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou