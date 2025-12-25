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
logger = setup_logging('fselection')
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr



def all_selections(X_train_num,X_test_num,y_train):
    try:
        logger.info(f'columns before constant technique{X_train_num.columns} -> {X_train_num.shape} -> {X_test_num.shape}')
        constant_reg = VarianceThreshold(0.0)
        constant_reg.fit(X_train_num)
        best_columns_constant = X_train_num.columns[constant_reg.get_support()]
        X_train_num = constant_reg.transform(X_train_num)
        X_test_num = constant_reg.transform(X_test_num)
        X_train_num = pd.DataFrame(X_train_num,columns=best_columns_constant)
        X_test_num = pd.DataFrame(X_test_num,columns=best_columns_constant)
        logger.info(f'columns after constant technique{X_train_num.columns} -> {X_train_num.shape} -> {X_test_num.shape}')
        logger.info(f'=========================================================')
        logger.info(f'columns before Quasi constant technique{X_train_num.columns} -> {X_train_num.shape} -> {X_test_num.shape}')
        quasi_constant_reg = VarianceThreshold(0.1)
        quasi_constant_reg.fit(X_train_num)
        best_columns_quasi_constant = X_train_num.columns[quasi_constant_reg.get_support()]
        X_train_num = quasi_constant_reg.transform(X_train_num)
        X_test_num = quasi_constant_reg.transform(X_test_num)
        X_train_num = pd.DataFrame(X_train_num, columns=best_columns_quasi_constant)
        X_test_num = pd.DataFrame(X_test_num, columns=best_columns_quasi_constant)
        logger.info(f'columns after Quasi constant technique{X_train_num.columns} -> {X_train_num.shape} -> {X_test_num.shape}')
        logger.info(f'===============================================================')
        sample_y_train = y_train.copy()
        sample_y_train = sample_y_train.map({'Good':1, 'Bad':0}).astype(int)
        c = []
        plt.figure(figsize=(5,3))
        for i in X_train_num.columns:
            c.append(pearsonr(X_train_num[i], sample_y_train))
        c = np.array(c)
        p_value = pd.Series(c[:,1],index = X_train_num.columns)
        p_value.sort_values(ascending = False ,inplace = True)
        X_train_num = X_train_num.drop(['DebtRatio_yeo_trim'],axis=1)
        X_test_num = X_test_num.drop(['DebtRatio_yeo_trim'], axis=1)
        return X_train_num, X_test_num





    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")

