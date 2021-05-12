from sklearn.metrics import accuracy_score,roc_auc_score
import numpy as np


def AUC(gt,pred,task):
    if task == 'binary-class':
        pred = pred[:,-1]
        return roc_auc_score(gt, pred)
    elif task == 'multi-label, binary-class':
        auc = 0
        for i in range(pred.shape[1]):
            label_auc = roc_auc_score(gt[:, i], pred[:, i])
            auc += label_auc
        return auc / pred.shape[1]
    else:
        auc = 0
        zero = np.zeros_like(gt)
        one = np.ones_like(gt)
        for i in range(pred.shape[1]):
            gt_binary = np.where(gt == i, one, zero)
            pred_binary = pred[:, i]
            auc += roc_auc_score(gt_binary, pred_binary)
        return auc / pred.shape[1]


def ACC(gt,pred,task,threshold=0.5):
    if task == 'multi-label, binary-class':
        zero = np.zeros_like(pred)
        one = np.ones_like(pred)
        y_pre = np.where(pred < threshold, zero, one)
        acc = 0
        for label in range(gt.shape[1]):
            label_acc = accuracy_score(gt[:, label], y_pre[:, label])
            acc += label_acc
        return acc / gt.shape[1]
    elif task == 'binary-class':
        y_pre = np.zeros_like(gt)
        for i in range(pred.shape[0]):
            y_pre[i] = (pred[i][-1] > threshold)
        return accuracy_score(gt, y_pre)
    else:
        y_pre = np.zeros_like(gt)
        for i in range(pred.shape[0]):
            y_pre[i] = np.argmax(pred[i])
        return accuracy_score(gt, y_pre)
