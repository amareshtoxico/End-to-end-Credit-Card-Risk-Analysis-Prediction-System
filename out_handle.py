import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sys
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import logging
from log_file import setup_logging
logger = setup_logging('out_handle')


def trimming(X_train_num,X_test_num):
    try:
        for i in X_train_num.columns:
            iqr = X_train_num[i].quantile(0.75) - X_train_num[i].quantile(0.25)
            upper_limit = X_train_num[i].quantile(0.75) + 1.5 * iqr
            lower_limit = X_train_num[i].quantile(0.25) - 1.5 * iqr
            X_train_num[i + "_trim"] = np.where(X_train_num[i] > upper_limit,upper_limit,
                                               np.where(X_train_num[i] < lower_limit,lower_limit,X_train_num[i]))
            X_test_num[i + "_trim"] = np.where(X_test_num[i] > upper_limit,upper_limit,
                                               np.where(X_test_num[i] < lower_limit, lower_limit, X_test_num[i]))
            X_train_num = X_train_num.drop([i],axis = 1)
            X_test_num = X_test_num.drop([i],axis = 1)
        logger.info(f"X_train_num_columns_names : {X_train_num.columns} -> {X_train_num.shape}")
        logger.info(f"X_test_num_columns_names : {X_test_num.columns} -> {X_test_num.shape}")

        return X_train_num,X_test_num
    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")
