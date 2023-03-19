# from src.components import data_ingestion # Path: src\components\data_transformation.py
import pandas as pd
import numpy as np 
import os
import sys # sys is used to get the system information like python version, os, etc
from dataclasses import dataclass #this dataclass is used to create a class with attributes and methods without any boilerplate code
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder # StandardScaler is used to scale the data and OneHotEncoder is used to encode the categorical data into numerical data 
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object # utils is used to save the preprocessor object

@dataclass # this dataclass is used to create a class with attributes and methods without any boilerplate code
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')
# data_paths=data_ingestion.DataIngestion().initiate_data_ingestion()

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This method is used to get the preprocessor object which is used to transform the data
        '''
        try:
            numerical_columns=['writing_score','reading_score']
            categorical_columns=[
                "gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"
            ]
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('std_scaler',StandardScaler())
                ])
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f'Categorical columns: {categorical_columns}')
            logging.info(f'Numerical columns: {numerical_columns}')

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns) 
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('train and test data readed successfully')
            logging.info('obtaining preprocessor object')
            preprocessor_obj=self.get_data_transformer_object()
            target_column_name="math_score"
            # numerical_columns=['writing_score','reading_score']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info(
                f"APPLYING PREPROCESSING ON TRAINIG DATAFRAME AND TESTING DATAFRAME"
            )

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
                ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info("Saved processing Object to artifact folder")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)