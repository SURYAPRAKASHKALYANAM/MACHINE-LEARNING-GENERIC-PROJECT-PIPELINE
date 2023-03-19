import os 
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as f:
            dill.dump(obj,f)

    except Exception as e:
        raise CustomException(e,sys)   
    
def evaluate_model(X_train,Y_train,X_test,Y_test,models:dict):
    try:
        model_report:dict={}
        for model_name,model in models.items():
            model.fit(X_train,Y_train)
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)
            train_model_score=r2_score(Y_train,y_train_pred)
            test_model_score=r2_score(Y_test,y_test_pred)
            model_report[model_name]=test_model_score
        return model_report

    except Exception as e:
        raise CustomException(e,sys)