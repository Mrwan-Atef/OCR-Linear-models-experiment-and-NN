from sklearn.metrics import precision_score, recall_score ,f1_score , accuracy_score

from sklearn.metrics import classification_report, confusion_matrix , ConfusionMatrixDisplay, roc_auc_score , roc_curve
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['acc', 'precision', 'recall', 'f1', 'class_report', 'conf_matrix', 'plot_confusion_matrix']
def acc (y_true, y_pred):
    return accuracy_score(y_true, y_pred)*100

def precision (y_true, y_pred):
    return precision_score(y_true, y_pred, average='weighted')*100

def recall (y_true, y_pred):
    return recall_score(y_true, y_pred, average='weighted')*100

def f1 (y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')*100
 
def class_report (y_true, y_pred):
   print(classification_report(y_true, y_pred))
   
def conf_matrix (y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(cm):
    classes =  range(10)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=15, pad=20)
    plt.xlabel('Prediction', fontsize=11)
    plt.ylabel('Actual', fontsize=11)
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.tick_top()
    plt.gca().figure.subplots_adjust(bottom=0.2)
    plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)

    plt.show()

