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
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
#    ind = linear_assignment(w.max() - w)
#    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    row_ind, col_ind = linear_sum_assignment(w.max() - w) #difference because we want to maximise the matching, e.g. minimize the cost (orginal problem)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = w[row_ind, col_ind].sum() / y_pred.size
    return accuracy, reassignment 
