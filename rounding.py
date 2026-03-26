
import numpy as np
__all__ = ['round_regression_to_classification']
def round_regression_to_classification(y_test_pred):
# round predictions to nearest integer and clamp to [0,9]
    y_test_pred_round = np.rint(y_test_pred).astype(int)
    y_test_pred_round = np.clip(y_test_pred_round, 0, 9)

    return y_test_pred_round


