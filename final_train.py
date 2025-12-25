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
logger = setup_logging('final_train')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import pickle

def lr(X_train, y_train, X_test, y_test):
    try:
        logger.info(f'Logistic Regression Training Started')
        reg = LogisticRegression()
        reg.fit(X_train, y_train)
        logger.info(f'Train Accuracy: {accuracy_score(y_train, reg.predict(X_train))}')
        logger.info(f'Test Accuracy: {accuracy_score(y_test, reg.predict(X_test))}')
        logger.info(f'Test Confusion matrix: {confusion_matrix(y_test, reg.predict(X_test))}')
        logger.info(f'Test Classification report: {classification_report(y_test, reg.predict(X_test))}')
        with open('credit_card.pkl', 'wb') as f:
            pickle.dump(reg, f)


        logger.info(f'Training and saving the model into pickle file completed')
        logger.info(f'Load the model from file and check the model is working or not')
        with open('credit_card.pkl', 'rb') as p:
            model = pickle.load(p)
        with open('standard_scaler.pkl', 'rb') as p1:
            scaler = pickle.load(p1)

        random_inputs = np.random.random((6, 2)).ravel()
        logger.info(random_inputs.ndim)
        random_inputs = random_inputs.reshape(1,-1)
        scaled_random_inputs = scaler.transform(random_inputs)
        result_from_model = model.predict(scaled_random_inputs)
        logger.info(f'Model prediction : {result_from_model}')
        if result_from_model[0] == 0:
            logger.info(f'Bad customer')
        else:
            logger.info(f'Good customer')

    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")