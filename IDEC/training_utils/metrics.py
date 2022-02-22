from __future__ import division, print_function
import numpy as np
#from linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment


#######################################################
# Evaluate Critiron
#######################################################


def cluster_acc(y_true, y_pred):
    #Adjusted from © dawnranger.
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
        reassignment dictionary 
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size

    true_labels_unique = np.unique(y_true)
    true_labels_upd = np.arange(len(true_labels_unique))
    true_label_dict = {int(true_labels_unique[i]): int(true_labels_upd[i]) for i in range(len(true_labels_unique))}
    true_label_reverse = {true_label_dict[key]:key for key in true_label_dict}

    #D = max(y_pred.max(), y_true.max()) + 1
    D = max(len(np.unique(y_pred)), len(true_labels_upd)) 
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], true_label_dict[int(y_true[i])]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w) #difference because we want to maximise the matching, e.g. minimize the cost (orginal problem)
    #reassignment = dict(zip(row_ind, col_ind))

    added_indexes = set(np.unique(col_ind))-set(true_labels_upd)
    for a_i in added_indexes:
        true_label_reverse[int(a_i)] = 10+np.max(true_labels_unique)

    reassignment = dict(zip(row_ind,[true_label_reverse[c] for c in col_ind]))
    accuracy = w[row_ind, col_ind].sum() / y_pred.size
    return accuracy, reassignment 
