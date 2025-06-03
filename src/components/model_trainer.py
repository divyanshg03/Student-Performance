from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
@dataclass
class ModelTrainerConfig:
    train_model_file_path: str = os.path.join('artifacts', 'model.pkl')
   
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            
            logging.info("Splitting training and testing data")
            
            
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=False),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'XGBRegressor': XGBRegressor()
            }

            model_report = {}
            
            for model_name, model in models.items():
                
                model.fit(X_train, y_train)
                
                y_train_pred = model.predict(X_train)
                
                y_pred = model.predict(X_test)
                
                train_model_score = r2_score(y_train,y_train_pred)
                test_model_score = r2_score(y_test, y_pred)
                model_report[model_name] = test_model_score
                logging.info(f"{model_name} R2 Score: {test_model_score}")

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found")

            logging.info(f"Best model: {best_model_name} with R2 Score: {best_model_score}")

            save_object(
                file_path = self.model_trainer_config.train_model_file_path,
                obj = {
                    'model': best_model
                }
            )
            predicted = best_model.predict(X_test)
            r2_score_value = r2_score(y_test, predicted)
            logging.info(f"R2 Score of the best model: {r2_score_value}")

            return r2_score_value

        except Exception as e:
            raise CustomException(e, sys)
