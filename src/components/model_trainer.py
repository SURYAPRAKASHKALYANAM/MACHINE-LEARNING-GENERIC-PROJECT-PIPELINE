import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.metrics import accuracy_score,r2_score
from sklearn.ensemble import (GradientBoostingRegressor,
                               AdaBoostRegressor,
                               RandomForestRegressor)
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,Y_train,X_test,Y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            # param_grid is used for hyper paramater tuning of model
            # param_grid = {
            #     'Linear Regression': {
            #         'fit_intercept': [True, False],
            #         'normalize': [True, False],
            #     },
            #     'Lasso': {
            #         'alpha': [0.1, 1.0, 10.0],
            #         'fit_intercept': [True, False],
            #         'normalize': [True, False],
            #     },
            #     'Ridge': {
            #         'alpha': [0.1, 1.0, 10.0],
            #         'fit_intercept': [True, False],
            #         'normalize': [True, False],
            #     },
            #     'K-Neighbors Regressor': {
            #         'n_neighbors': [5, 10, 20],
            #         'weights': ['uniform', 'distance'],
            #         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            #     },
            #     'Decision Tree': {
            #         'max_depth': [None, 5, 10],
            #         'min_samples_split': [2, 5, 10],
            #         'min_samples_leaf': [1, 2, 4],
            #     },
            #     'Random Forest Regressor': {
            #         'n_estimators': [100, 500],
            #         'max_depth': [None, 5, 10],
            #         'min_samples_split': [2, 5, 10],
            #         'min_samples_leaf': [1, 2, 4],
            #     },
            #     'XGBRegressor': {
            #         'n_estimators': [100, 500],
            #         'max_depth': [None, 5, 10],
            #         'learning_rate': [0.01, 0.1, 0.5],
            #     },
            #     'CatBoosting Regressor': {
            #         'n_estimators': [100, 500],
            #         'max_depth': [None, 5, 10],
            #         'learning_rate': [0.01, 0.1, 0.5],
            #         'silent': [True, False],
            #     },
            #     'AdaBoost Regressor': {
            #         'n_estimators': [50, 100, 500],
            #         'learning_rate': [0.01, 0.1, 0.5],
            #         'loss': ['linear', 'square', 'exponential'],
            #     },
            # }


            logging.info("Training models")
            model_report:dict=evaluate_model(X_train,Y_train,X_test,Y_test,models=models)

            # to get the best model score from dict of model scores
            best_model_score=max(sorted(model_report.values()))

            # to get the best model name from dict of model names
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model=models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("Model score is less than 60% No best Model")
            logging.info("BEST MODEL FOUND")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)
            logging.info("Model saved successfully")
            predicted_values=best_model.predict(X_test)
            r2_scoree=r2_score(Y_test,predicted_values)
            return  r2_scoree
            


        except Exception as e:
            raise CustomException(e)    