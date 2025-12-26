import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utlis import save_obj
from src.utlis import evalute_models
@dataclass
class ModelTrainerConfig:
    trainded_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Splitting Training and test input data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "KNeighborsRegressor" : KNeighborsRegressor(),
                "XGB Regressor" : XGBRegressor(),
                "Adaboost Regressor ": AdaBoostRegressor(),
                "linear Regressor " :LinearRegression()
            }
            params = {
                "Random Forest" : {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [None, 5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": [None, "sqrt", "log2"]
                },
                "Decision Tree" : {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 5],
                    "max_features": [None, "sqrt", "log2"]
                },
                "Gradient Boosting" : {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.6, 0.8, 1.0],
                    "max_depth": [3, 4, 5],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"]
                },
                "K-Neighbours Regressor" : {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan", "minkowski"],
                    "p": [1, 2]   
                },
                "XGB Regressor" : {
                    "n_estimators": [200, 400, 600],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7, 9],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0],
                    "reg_alpha": [0, 0.01, 0.1],
                    "reg_lambda": [0.5, 1, 2]
                },
                "Adaboost Regressor " : {
                    "n_estimators": [50, 100, 200, 400],
                    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                    "loss": ["linear", "square", "exponential"]
                }

                
            }
            model_report:dict = evalute_models(x_train,y_train,x_test,y_test,models,params)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            #To get best model name from dict

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best Model found")
            
            logging.info(f"Best model is found : {best_model_name}")

            save_obj(
                file_path=self.model_trainer_config.trainded_model_file_path,
                obj= best_model
            )
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)

            return r2_square
        

        except Exception as e:
            raise CustomException(e,sys)







