def IoUScore(y_pred, y_true):
    epsilon = 1e-6
    y_pred_bin = (torch.sigmoid(y_pred) > 0.5).float()
    y_true_bin = (y_true > 0.5).float()
    intersection = (y_pred_bin * y_true_bin).sum((1,2,3))
    union = y_pred_bin.sum((1,2,3)) + y_true_bin.sum((1,2,3)) - intersection
    return (intersection / (union + epsilon)).mean().item()

def DiceScore(y_pred, y_true):
    iou = IoUScore(y_pred, y_true)
    return (2 * iou) / (1 + iou + 1e-6)

def PrecisionScore(y_pred, y_true):
    epsilon = 1e-6
    y_pred_bin = (torch.sigmoid(y_pred) > 0.5).float()
    y_true_bin = (y_true > 0.5).float()
    tp = (y_pred_bin * y_true_bin).sum((1,2,3))
    return (tp / (y_pred_bin.sum((1,2,3)) + epsilon)).mean().item()

def RecallScore(y_pred, y_true):
    epsilon = 1e-6
    y_pred_bin = (torch.sigmoid(y_pred) > 0.5).float()
    y_true_bin = (y_true > 0.5).float()
    tp = (y_pred_bin * y_true_bin).sum((1,2,3))
    return (tp / (y_true_bin.sum((1,2,3)) + epsilon)).mean().item()

def DiceScore(y_pred, y_true):
    epsilon = 1e-6
    y_pred_bin = (torch.sigmoid(y_pred) > 0.5).float()
    y_true_bin = (y_true > 0.5).float()

    intersection = (y_pred_bin * y_true_bin).sum((1, 2, 3))
    dice = (2. * intersection + epsilon) / (y_pred_bin.sum((1, 2, 3)) + y_true_bin.sum((1, 2, 3)) + epsilon)
    return dice.mean().item()

def IoUScore(y_pred, y_true):
    epsilon = 1e-6
    y_pred_bin = (torch.sigmoid(y_pred) > 0.5).float()
    y_true_bin = (y_true > 0.5).float()

    intersection = (y_pred_bin * y_true_bin).sum((1, 2, 3))
    union = y_pred_bin.sum((1, 2, 3)) + y_true_bin.sum((1, 2, 3)) - intersection
    return (intersection / (union + epsilon)).mean().item()
