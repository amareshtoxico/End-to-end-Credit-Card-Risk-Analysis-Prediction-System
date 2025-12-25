import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import logging
from log_file import setup_logging
logger = setup_logging('random_sample_tech')


def random_tech(X_train,X_test):
    try:
        logger.info(f"Before Filling Null values Training data: {X_train.shape}")
        logger.info(f"Before Filling Null values Testing data: {X_test.shape}")
        for i in X_train.columns:
            if X_train[i].isnull().sum() > 0:
                X_train[i+'_replaced'] = X_train[i].copy()
                X_test[i+"_replaced"] = X_test[i].copy()
                s = X_train.dropna().sample(X_train[i].isnull().sum(),random_state=42)
                s1 = X_test.dropna().sample(X_test[i].isnull().sum(), random_state=42)
                s.index = X_train[X_train[i].isnull()].index
                s1.index = X_test[X_test[i].isnull()].index
                X_train.loc[X_train[i].isnull() , i+'_replaced']= s
                X_test.loc[X_test[i].isnull(), i+'_replaced'] = s1
                X_train = X_train.drop([i],axis=1)
                X_test = X_test.drop([i],axis=1)
        logger.info(f"After Filling Null values : {X_train.shape}")
        logger.info(f"After Filling Null values : {X_test.shape}")
        logger.info(f"Training data null values : {X_train.isnull().sum()}")
        logger.info(f"testing data null values : {X_test.isnull().sum()}")
        return X_train,X_test

    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")
