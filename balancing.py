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
logger = setup_logging('balancing')
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler # z_score = (x - mean ) / std
# from train_models_ML import common
from final_train import lr
import pickle


def bala_data(training_ind_data,y_train,testing_ind_data,y_test):
    try:
        logger.info(f' Before Number of rows for Good customers : {sum(y_train == 1)}')
        logger.info(f' Before Number of rows for Bad customers : {sum(y_train == 0)}')

        sm_reg = SMOTE(random_state=42)
        training_ind_data_bal,y_train_bal = sm_reg.fit_resample(training_ind_data,y_train)

        logger.info(f'After Number of rows for Good customers : {sum(y_train_bal == 1)}')
        logger.info(f'After Number of rows for Bad customers : {sum(y_train_bal == 0)}')

        logger.info("================= Before Scaling ===========================")
        logger.info(f'{training_ind_data_bal.sample(7)}')
        sc = StandardScaler()
        sc.fit(training_ind_data_bal)
        training_ind_data_bal_sc = sc.transform(training_ind_data_bal)
        testing_ind_data_sc = sc.transform(testing_ind_data)
        with open('standard_scaler.pkl', 'wb') as f1:
            pickle.dump(sc, f1)
        logger.info("================= After Scaling ===========================")
        logger.info(f'{training_ind_data_bal_sc}')

        # common(training_ind_data_bal_sc,y_train_bal,testing_ind_data_sc,y_test)

        lr(training_ind_data_bal_sc,y_train_bal,testing_ind_data_sc,y_test)

    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")