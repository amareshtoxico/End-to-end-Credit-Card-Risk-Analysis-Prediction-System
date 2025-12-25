'''
In this file we are going to read data and make connections based on requirements
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.filterwarnings('ignore')
import logging
from log_file import setup_logging
logger = setup_logging('main')
from sklearn.model_selection import train_test_split
from random_sample_tech import random_tech
import sys
from yeojohnson_tech import vt
from out_handle import trimming
from fselection import all_selections
from cat_to_n import changing_data_to_num
from balancing import bala_data

class CREDIT_CARD_INFO:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            logger.info(f"The Data loaded Successfully ")
            logger.info(f"we have : {self.df.shape[0]} Rows and {self.df.shape[1]} Columns")
            self.df = self.df.drop([150000,150001],axis=0)
            self.df = self.df.drop(['MonthlyIncome.1'],axis=1)
            logger.info(f"{self.df.isnull().sum()}")
            self.df['NumberOfDependents'] = pd.to_numeric(self.df['NumberOfDependents'])
            logger.info(f"MonthlyIncome columns data type : {self.df['MonthlyIncome'].dtype}")
            logger.info(f"NumberOfDependents columns data type : {self.df['NumberOfDependents'].dtype}")
            self.X = self.df.iloc[: , :-1] # independent
            self.y = self.df.iloc[: , -1] # dependent
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            logger.info(f"Training Data size : {self.X_train.shape}")
            logger.info(f"Test Data Size : {self.X_test.shape}")
        except Exception as e:
            er_ty,er_msg,er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")



    def missing_values(self):
        try:
            self.X_train,self.X_test = random_tech(self.X_train,self.X_test)
            logger.info('Missing values completed Successfully')
            self.X_train_num = self.X_train.select_dtypes(exclude='object')
            self.X_train_cat = self.X_train.select_dtypes(include='object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')
            self.X_test_cat = self.X_test.select_dtypes(include='object')
            logger.info(f"=======Training Numerical columns==============")
            logger.info(f'{self.X_train_num.columns}')
            logger.info(f"=======Training Categorical columns==============")
            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f"=======Testing Numerical columns==============")
            logger.info(f'{self.X_test_num.columns}')
            logger.info(f"=======Testing Categorical columns==============")
            logger.info(f'{self.X_test_cat.columns}')
        except Exception as e:
            er_ty,er_msg,er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")

    def varibale_transformation_tech(self):
        try:
            self.X_train_num,self.X_test_num = vt(self.X_train_num,self.X_test_num)
            logger.info(f"{self.X_train_num.isnull().sum()}")
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")

    def outliers_handle(self):
        try:
            self.X_train_num,self.X_test_num = trimming(self.X_train_num,self.X_test_num)
            logger.info("Confirm Again In Main File")
            logger.info(f"X_train_num_column_Names : {self.X_train_num.columns} -> {self.X_train_num.shape}")
            logger.info(f"X_test_num_column_Names : {self.X_test_num.columns} -> {self.X_test_num.shape}")
            logger.info(f"Outliers handled SuccessFully")

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")

    def feature_selection(self):
        try:
            logger.info(f'before feature selection')
            logger.info(f'{self.X_train_num.shape}')
            logger.info(f'{self.X_test_num.shape}')
            self.X_train_num,self.X_test_num = all_selections(self.X_train_num,self.X_test_num,self.y_train)
            logger.info(f'after feature selection')
            logger.info(f'{self.X_train_num.shape}')
            logger.info(f'{self.X_test_num.shape}')
            logger.info(f'{self.X_train_cat.shape}')
            logger.info(f'{self.X_test_cat.shape}')

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")

    def Cat_to_num(self):
        try:
            logger.info("Before Converting into Numerical")
            logger.info(f"{self.X_train_cat.columns}")
            logger.info(f"{self.X_train_cat.isnull().sum()}")
            logger.info("Before Converting into Numerical")
            logger.info(f"{self.X_test_cat.columns}")
            logger.info(f"{self.X_test_cat.isnull().sum()}")

            self.X_train_cat, self.X_test_cat = changing_data_to_num(self.X_train_cat, self.X_test_cat)

            logger.info("After Converting into Numerical")
            logger.info(f"{self.X_train_cat.columns}")
            logger.info(f"{self.X_train_cat.isnull().sum()}")
            logger.info("After Converting into Numerical")
            logger.info(f"{self.X_test_cat.columns}")
            logger.info(f"{self.X_test_cat.isnull().sum()}")
            self.X_train_num.reset_index(drop=True, inplace=True)
            self.X_train_cat.reset_index(drop=True, inplace=True)

            self.X_test_num.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)

            self.training_data = pd.concat([self.X_train_num, self.X_train_cat], axis=1)
            self.testing_data = pd.concat([self.X_test_num, self.X_test_cat], axis=1)

            logger.info(f"Final Training data")
            logger.info(f"{self.training_data.columns}")
            logger.info(f"{self.training_data.sample(10)}")
            logger.info(f"{self.training_data.isnull().sum()}")

            logger.info(f"Final Testing data")
            logger.info(f"{self.testing_data.columns}")
            logger.info(f"{self.testing_data.sample(10)}")
            logger.info(f"{self.testing_data.isnull().sum()}")

            logger.info(f'{self.training_data.shape}')
            logger.info(f'{self.testing_data.shape}')
            logger.info(f'{self.y_train.shape}')


        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")

    def data_balancing(self):
        try:
            self.y_train = self.y_train.map({'Good' : 1,'Bad' : 0}).astype(int)
            self.y_test = self.y_test.map({'Good': 1, 'Bad': 0}).astype(int)
            bala_data(self.training_data,self.y_train,self.testing_data,self.y_test)
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")


if __name__ == "__main__":
    try:
        data = 'creditcard.csv'
        obj = CREDIT_CARD_INFO(data)
        obj.missing_values()
        obj.varibale_transformation_tech()
        obj.outliers_handle()
        obj.feature_selection()
        obj.Cat_to_num()
        obj.data_balancing()
    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")