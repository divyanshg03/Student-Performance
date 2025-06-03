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
from sklearn.model_selection import GridSearchCV

import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
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

            params = {
                'LinearRegression': {},
                
                'RandomForestRegressor': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],    
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'GradientBoostingRegressor': {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'AdaBoostRegressor': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5]
                },
                'CatBoostRegressor': {
                    'iterations': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [4, 6, 8]
                },
                'DecisionTreeRegressor': {
                    'max_depth': [5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'KNeighborsRegressor': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree']
                },
                'XGBRegressor': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
              # Use evaluate_models function for hyperparameter tuning
            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                         models=models, params=params)

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            
            # Get the best model and fit it with best parameters
            best_model = models[best_model_name]
            best_params = params[best_model_name]
            
            if best_params:  # Only if parameters are provided
                gs = GridSearchCV(best_model, best_params, cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                logging.info(f"Best parameters for {best_model_name}: {gs.best_params_}")
            else:  # For models without hyperparameters (like LinearRegression)
                best_model.fit(X_train, y_train)
                logging.info(f"Fitted {best_model_name} without hyperparameter tuning")

            if best_model_score < 0.6:
                raise CustomException("No suitable model found")

            save_object(self.model_trainer_config.train_model_file_path, best_model)
            logging.info(f"Best model: {best_model_name} with R2 Score: {best_model_score}")
            predicted = best_model.predict(X_test)
            r2_score_value = r2_score(y_test, predicted)
            logging.info(f"R2 Score of the best model: {r2_score_value}")

            return r2_score_value

        except Exception as e:
            raise CustomException(e, sys)
        
