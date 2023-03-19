import os
import sys
from  src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass #this dataclass is used to create a class with attributes and methods without any boilerplate code    
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact','train.csv')
    test_data_path: str=os.path.join('artifact','test.csv')
    raw_data_path: str=os.path.join('artifact','data.csv')

class DataIngestion:
    # this class is used to ingest the data from the source and split the data into train and test 
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    # this method is used to split the data into train and test  
    def initiate_data_ingestion(self):
        try:
            logging.info('Data ingestion started')
            df=pd.read_csv("notebook\data\stud.csv")
            logging.info('DATA LOADED SUCCESSFULLY FROM SOURCE TO DATAFRAME') 
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('TRAIN TEST SPLIT INITIATED')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('TRAIN TEST SPLIT COMPLETED SUCCESSFULLY')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error('Error occured in data ingestion')
            raise CustomException(e,sys)        

if __name__ == '__main__':
    train_data,test_data=DataIngestion().initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))
