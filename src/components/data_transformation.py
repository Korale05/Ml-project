import sys 
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utlis import save_obj
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts",'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation based on different data

        """
        try:
            cat_feature = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
                            'test_preparation_course']
            num_feature = ['reading_score', 'writing_score']


            num_pipline = Pipeline(
                steps=[
                ("Imputer",SimpleImputer(strategy='median')),
                ("StandardSclaer",StandardScaler())
                ]
            )

            cat_pipline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy='most_frequent')),
                    ("one hot encoder",OneHotEncoder())
                ]
            )
            

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipline,num_feature),
                    ("cat_pipline",cat_pipline,cat_feature)
                ]
            )

            logging.info("Transformation pipline is completed")


            return preprocessor

        except Exception as e:
            logging.info(f"Error is happen : {e}")
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            target_columns_name='math_score'
            num_feature = ['reading_score', 'writing_score','math_score']

            input_feature_train_df = train_df.drop(columns=[target_columns_name],axis=1)
            target_feature_train_df = train_df[target_columns_name]

            input_feature_test_df = test_df.drop(columns=[target_columns_name],axis=1)
            target_feature_test_df = test_df[target_columns_name]

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object.")

            save_obj(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info("Exception is happen")
            raise CustomException(e,sys)