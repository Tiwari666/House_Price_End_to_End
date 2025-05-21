import os
import sys
#  treat the project root (LR_BankLoan) as the root module — so from src.exception will now work.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# from src.components.model_trainer import ModelTrainer
# from src.components.model_trainer import ModelTrainerConfig

# Step 1: Define where to save files (using @dataclass)
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

# Step 2: Create a class for data ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    # Step 3: Define the ingestion process    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            # use the actual data file: loan_data.csv
            df = pd.read_csv('notebook/data/raw/loan_data.csv')

            logging.info("Read the Dataset as pandas dataframe")

            # Create directories/folders if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved")

            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Train and test data saved")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )

        except Exception as e:
            raise CustomException(e, sys) from e

# Step 4: Test the DataIngestion class / Run this step and connect next steps
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path, test_data_path, raw_data_path = data_ingestion.initiate_data_ingestion()
################################################################################
    # # Step 5: Call the DataTransformation class
    data_transformation = DataTransformation()
    data_transformation_config = DataTransformationConfig()

    # # Step 6: Call the ModelTrainer class
    model_trainer = ModelTrainer()
    model_trainer_config = ModelTrainerConfig()

    # # Call the data transformation and model trainer methods to Make sure model receives transformed data
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    print(model_trainer.initiate_model_trainer(train_arr, test_arr))



############################################################
# Code to run the data_ingestion.py file:
# my current loaction is : C:\Users\naren\Desktop\New_DS\LR_BankLoan>
# python src/components/data_ingestion.py
# 


