import numpy as np


def accuracy(ground_truth, predictions):
    assert len(ground_truth) == len(predictions)

    if type(ground_truth) is not np.ndarray:
        ground_truth = np.array(ground_truth)

    if type(predictions) is not np.ndarray:
        predictions = np.array(predictions)

    return np.sum(ground_truth == predictions) / len(ground_truth)


def confusion_matrix(ground_truth, predictions):
    assert len(ground_truth) == len(predictions)
    assert min(ground_truth) == 0
    assert max(ground_truth) == len(np.unique(ground_truth)) - 1

    if type(ground_truth) is not np.ndarray:
        ground_truth = np.array(ground_truth)

    if type(predictions) is not np.ndarray:
        predictions = np.array(predictions)

    n_classes = len(np.unique(ground_truth))
    result = np.empty((n_classes, ) * 2, np.int64)

    for i in range(n_classes):
        for j in range(n_classes):
            result[i][j] = sum(((ground_truth == i) & (predictions == j)))

    return result


def class_accuracies(ground_truth, predictions):
    assert len(ground_truth) == len(predictions)

    if type(ground_truth) is not np.ndarray:
        ground_truth = np.array(ground_truth)

    if type(predictions) is not np.ndarray:
        predictions = np.array(predictions)

    result = {}

    for label in np.unique(ground_truth):
        indices = (ground_truth == label)
        result[label] = accuracy(ground_truth[indices], predictions[indices])

    return result


def class_recalls(ground_truth, predictions):
    assert len(ground_truth) == len(predictions)

    if type(ground_truth) is not np.ndarray:
        ground_truth = np.array(ground_truth)

    if type(predictions) is not np.ndarray:
        predictions = np.array(predictions)

    cm = np.array(confusion_matrix(ground_truth, predictions))
    result = {}

    for label in np.unique(ground_truth):
        result[label] = cm[label][label] / sum(cm[label])

    return result


def average_accuracy(ground_truth, predictions):
    assert len(ground_truth) == len(predictions)

    return np.mean(list(class_accuracies(ground_truth, predictions).values()))


def class_balance_accuracy(ground_truth, predictions):
    assert len(ground_truth) == len(predictions)

    n_classes = len(np.unique(ground_truth))
    cm = confusion_matrix(ground_truth, predictions)
    result = 0

    for i in range(n_classes):
        sum_cm_i_j = sum([cm[i][j] for j in range(n_classes)])
        sum_cm_j_i = sum([cm[j][i] for j in range(n_classes)])

        result += cm[i][i] / max(sum_cm_i_j, sum_cm_j_i)

    result /= n_classes

    return result


def geometric_average_of_recall(ground_truth, predictions):
    assert len(ground_truth) == len(predictions)

    recalls = class_recalls(ground_truth, predictions).values()
    result = 1.0

    for recall in recalls:
        result *= recall

    result = result ** (1 / len(recalls))

    return result
