import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            model_name = list(models.keys())[i]
            
            # Handle models with and without hyperparameters
            if param:  # If hyperparameters exist
                gs = GridSearchCV(model, param, cv=3, n_jobs=-1, verbose=1)
                gs.fit(X_train, y_train)
                
                logging.info(f"Best parameters for {model_name}: {gs.best_params_}")
                
                # Make predictions using the best estimator
                y_train_pred = gs.predict(X_train)
                y_test_pred = gs.predict(X_test)
            else:  # If no hyperparameters (like LinearRegression)
                model.fit(X_train, y_train)
                
                logging.info(f"No hyperparameters to tune for {model_name}")
                
                # Make predictions using the fitted model
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

            # Calculate R2 scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            
            print(f"{model_name} - Train R2: {train_model_score:.4f}, Test R2: {test_model_score:.4f}")
            
        return report
    
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)