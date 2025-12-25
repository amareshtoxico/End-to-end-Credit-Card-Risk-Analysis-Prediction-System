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
logger = setup_logging('Cat_to_n')
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder

def changing_data_to_num(X_train_cat,X_test_cat):
    try:
        one_hot = OneHotEncoder(drop='first')
        # since gender and region columns are nominal we are going to apply OnehotEnoder
        one_hot.fit(X_train_cat[['Gender', 'Region']])
        result = one_hot.transform(X_train_cat[['Gender', 'Region']]).toarray()
        result1 = one_hot.transform(X_test_cat[['Gender', 'Region']]).toarray()
        f = pd.DataFrame(data=result, columns=one_hot.get_feature_names_out())
        f1 = pd.DataFrame(data=result1, columns=one_hot.get_feature_names_out())
        X_train_cat.reset_index(drop=True, inplace=True)
        f.reset_index(drop=True, inplace=True)
        X_test_cat.reset_index(drop=True, inplace=True)
        f1.reset_index(drop=True, inplace=True)
        X_train_cat = pd.concat([X_train_cat, f], axis=1)
        X_train_cat = X_train_cat.drop(['Gender', 'Region'], axis=1)
        X_test_cat = pd.concat([X_test_cat, f1], axis=1)
        X_test_cat = X_test_cat.drop(['Gender', 'Region'], axis=1)
        logger.info(f"{X_train_cat.columns}")
        logger.info(f"{X_train_cat.sample(10)}")
        logger.info(f"{X_test_cat.columns}")
        logger.info(f"{X_test_cat.sample(10)}")
        logger.info('================Odinal Encoding==========================')
        od = OrdinalEncoder()
        od.fit(X_train_cat[['Rented_OwnHouse', 'Occupation', 'Education']])
        r1 = od.transform(X_train_cat[['Rented_OwnHouse', 'Occupation', 'Education']])
        r2 = od.transform(X_test_cat[['Rented_OwnHouse', 'Occupation', 'Education']])
        c_names = od.get_feature_names_out()
        c_names = c_names + '_Od'
        g1 = pd.DataFrame(data=r1, columns=c_names)
        g2 = pd.DataFrame(data=r2, columns=c_names)
        X_train_cat.reset_index(drop=True, inplace=True)
        g1.reset_index(drop=True, inplace=True)
        X_train_cat = pd.concat([X_train_cat, g1], axis=1)
        X_train_cat = X_train_cat.drop(['Rented_OwnHouse', 'Occupation', 'Education'], axis=1)

        X_test_cat.reset_index(drop=True, inplace=True)
        g2.reset_index(drop=True, inplace=True)
        X_test_cat = pd.concat([X_test_cat, g2], axis=1)
        X_test_cat = X_test_cat.drop(['Rented_OwnHouse', 'Occupation', 'Education'], axis=1)

        logger.info(f"{X_train_cat.columns}")
        logger.info(f"{X_train_cat.sample(10)}")
        logger.info(f"{X_test_cat.columns}")
        logger.info(f"{X_test_cat.sample(10)}")

        logger.info(f"{X_train_cat.isnull().sum()}")
        logger.info(f"{X_test_cat.isnull().sum()}")

        return X_train_cat, X_test_cat
    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")

