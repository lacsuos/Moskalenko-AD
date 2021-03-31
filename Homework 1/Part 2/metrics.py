import numpy as np

file = np.loadtxt('HW2_labels.txt', delimiter=',')
y_pred, y_true = file[:, :2], file[:, -1]


def confusion_matrix(y_true, y_predict): # Матрица несоответствий
    dim = y_predict.shape[1]
    y_predict = np.argmax(y_predict, axis=1)
    matrix = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            matrix[i, j] = ((y_true == i) * (y_predict == j)).sum()
    return matrix


def calculate_coeffs(y_true, y_predict): # Рассчет коэфициентов
    conf_matrix = confusion_matrix(y_true, y_predict)
    total = y_true.shape[0]
    if conf_matrix.shape[0] == 2:
        TP = conf_matrix[0, 0]
        FP = conf_matrix[0, 1]
        FN = conf_matrix[1, 0]
        TN = conf_matrix[1, 1]

    else:
        TP = np.diag(conf_matrix)
        FN = np.array([conf_matrix[i].sum() - TP[i] for i in range(TP.shape[0])])
        FP = np.array([conf_matrix[:, i].sum() - TP[i] for i in range(TP.shape[0])])
        TN = total - TP - FP - FN
    return TP, TN, FP, FN


def accuracy_score(y_true, y_predict, percent=None):
    if percent == None:
        percent = 50
    number = int(y_predict.shape[0] * percent / 100)
    assert number > 0, 'Выборка не включает ни одного элемента'
    conf_matrix = confusion_matrix(y_true[:number], y_predict[:number])
    total = y_predict[:number].shape[0]
    score = np.diag(conf_matrix).sum() / total
    return np.nan_to_num(score)


def precision_score(y_true, y_predict, percent=None):
    if percent == None:
        percent = 50
    number = int(y_predict.shape[0] * percent / 100)
    assert number > 0, 'Выборка не включает ни одного элемента'
    TP, TN, FP, FN = calculate_coeffs(y_true[:number], y_predict[:number])
    score = TP / (TP + FP)
    return np.nan_to_num(score)


def recall_score(y_true, y_predict, percent=None):
    if percent == None:
        percent = 50
    number = int(y_predict.shape[0] * percent / 100)
    assert number > 0, 'Выборка не включает ни одного элемента'
    TP, TN, FP, FN = calculate_coeffs(y_true[:number], y_predict[:number])
    score = TP / (TP + FN)
    return np.nan_to_num(score)


def f1_score(y_true, y_predict, percent=None):
    if percent == None:
        percent = 50
    number = int(y_predict.shape[0] * percent / 100)
    assert number > 0, 'Выборка не включает ни одного элемента'
    precision = precision_score(y_true, y_predict, percent)
    recall = recall_score(y_true, y_predict, percent)
    score = 2 * precision * recall / (precision + recall)
    return np.nan_to_num(score)


def lift_score(y_true, y_predict, percent=None):
    if percent == None:
        percent = 50
    number = int(y_predict.shape[0] * percent / 100)
    assert number > 0, 'Выборка не включает ни одного элемента'
    TP, TN, FP, FN = calculate_coeffs(y_true[:number], y_predict[:number])
    total = y_true[:number].shape[0]
    precision = precision_score(y_true, y_predict, percent)
    score = total * precision / (TP + FN)
    return np.nan_to_num(score)


print(accuracy_score(y_true, y_pred))
print(precision_score(y_true, y_pred))
print(recall_score(y_true, y_pred))
print(lift_score(y_true, y_pred))
print(f1_score(y_true, y_pred))

# Percent реализована как доля выборки, на котором мы проводим рассчет метрики. Делать ее как порог вероятности \
# бесмыссленно, так как в задаче мультиклассвой классификации может произойти, так что ни один объект ни на одном \
# классе не пройдет заданный порог вероятности или же, наоборот, на объекте может быть пройдено по порогу сразу \
# несколько классов.
