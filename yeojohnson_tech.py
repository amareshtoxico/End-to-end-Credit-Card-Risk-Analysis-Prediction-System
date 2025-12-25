
import sys
import warnings
warnings.filterwarnings('ignore')
import logging
from log_file import setup_logging
logger = setup_logging('yeojohnson_tech')
from scipy import stats


def vt(X_train_num,X_test_num):
    try:
        for i in X_train_num.columns:
            X_train_num[i + "_yeo"],alpha = stats.yeojohnson(X_train_num[i])
            X_test_num[i + "_yeo"],alpha = stats.yeojohnson(X_test_num[i])
            X_train_num = X_train_num.drop([i],axis=1)
            X_test_num = X_test_num.drop([i], axis=1)
        logger.info(f"{X_train_num.columns}")
        return X_train_num,X_test_num
    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")




