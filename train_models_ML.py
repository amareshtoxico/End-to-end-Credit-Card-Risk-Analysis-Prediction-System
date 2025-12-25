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
logger = setup_logging('train_models_ML')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score,roc_curve


def knn(X_train,y_train,X_test,y_test):
    try:
        knn_reg = KNeighborsClassifier(n_neighbors=5)
        knn_reg.fit(X_train,y_train)
        logger.info(f'Test Accuracy KNN : {accuracy_score(y_test,knn_reg.predict(X_test))}')
        logger.info(f'Test Confusion matrix KNN : {confusion_matrix(y_test, knn_reg.predict(X_test))}')
        logger.info(f'Test classification report KNN : {classification_report(y_test, knn_reg.predict(X_test))}')
        global knn_pred
        knn_pred = knn_reg.predict_proba(X_test)[:,1]

    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")

def nb(X_train,y_train,X_test,y_test):
    try:
        nb_reg = GaussianNB()
        nb_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy NB : {accuracy_score(y_test, nb_reg.predict(X_test))}')
        logger.info(f'Test Confusion matrix NB : {confusion_matrix(y_test, nb_reg.predict(X_test))}')
        logger.info(f'Test classification report NB : {classification_report(y_test, nb_reg.predict(X_test))}')
        global nb_pred
        nb_pred = nb_reg.predict_proba(X_test)[:, 1]
    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")

def lr(X_train,y_train,X_test,y_test):
    try:
        lr_reg = LogisticRegression()
        lr_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy LR : {accuracy_score(y_test, lr_reg.predict(X_test))}')
        logger.info(f'Test Confusion matrix LR : {confusion_matrix(y_test, lr_reg.predict(X_test))}')
        logger.info(f'Test classification report LR : {classification_report(y_test, lr_reg.predict(X_test))}')
        global lr_pred
        lr_pred = lr_reg.predict_proba(X_test)[:, 1]
    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")

def dt(X_train,y_train,X_test,y_test):
    try:
        dt_reg = DecisionTreeClassifier(criterion='entropy')
        dt_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy DT : {accuracy_score(y_test, dt_reg.predict(X_test))}')
        logger.info(f'Test Confusion matrix DT : {confusion_matrix(y_test, dt_reg.predict(X_test))}')
        logger.info(f'Test classification report DT : {classification_report(y_test, dt_reg.predict(X_test))}')
        global dt_pred
        dt_pred = dt_reg.predict_proba(X_test)[:, 1]
    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")

def rf(X_train,y_train,X_test,y_test):
    try:
        rf_reg = RandomForestClassifier(criterion='entropy',n_estimators=3)
        rf_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy RF : {accuracy_score(y_test, rf_reg.predict(X_test))}')
        logger.info(f'Test Confusion matrix RF : {confusion_matrix(y_test, rf_reg.predict(X_test))}')
        logger.info(f'Test classification report RF : {classification_report(y_test, rf_reg.predict(X_test))}')
        global rf_pred
        rf_pred = rf_reg.predict_proba(X_test)[:, 1]
    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")

def auc_roc(X_train,y_train,X_test,y_test):
    try:
        knn_fpr,knn_tpr,knn_threshold = roc_curve(y_test, knn_pred)
        nb_fpr, nb_tpr, nb_threshold = roc_curve(y_test, nb_pred)
        lr_fpr, lr_tpr, lr_threshold = roc_curve(y_test, lr_pred)
        dt_fpr, dt_tpr, dt_threshold = roc_curve(y_test, dt_pred)
        rf_fpr, rf_tpr, rf_threshold = roc_curve(y_test, rf_pred)

        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(knn_fpr,knn_tpr,label="KNN")
        plt.plot(nb_fpr, nb_tpr, label="navie bayes")
        plt.plot(lr_fpr, lr_tpr, label="Logistic Regression")
        plt.plot(dt_fpr, dt_tpr, label="Decision Tree")
        plt.plot(rf_fpr, rf_tpr, label="Random Forest")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve - ALL Models")
        plt.legend(loc=0)
        plt.show()

    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")



def common(X_train,y_train,X_test,y_test):
    try:
        logger.info(f'================ kNN ========================')
        knn(X_train, y_train, X_test, y_test)

        logger.info(f'================ NB ========================')
        nb(X_train, y_train, X_test, y_test)

        logger.info(f'================ LR ========================')
        lr(X_train, y_train, X_test, y_test)

        logger.info(f'================ DT ========================')
        dt(X_train, y_train, X_test, y_test)

        logger.info(f'================ RF ========================')
        rf(X_train, y_train, X_test, y_test)

        logger.info(f'================ AUC & ROC ========================')
        auc_roc(X_train, y_train, X_test, y_test)


    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")


