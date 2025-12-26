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
                "linear Regressor " :LinearRegression(),
                "K-Neighbours Regressor" : KNeighborsRegressor(),
                "XGB Regressor" : XGBRegressor(),
                "Adaboost Regressor ": AdaBoostRegressor()
            }
            model_report:dict = evalute_models(x_train,y_train,x_test,y_test,models)

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







