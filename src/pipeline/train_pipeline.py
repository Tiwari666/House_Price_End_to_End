#  train_pipeline.py (inside src/pipeline/)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

if __name__ == "__main__":
    try:
        logging.info("Starting training pipeline...")

        # 1. Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path,_ = ingestion.initiate_data_ingestion()

        # 2. Data Transformation
        transformer = DataTransformation()
        train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path)

        # 3. Model Training
        trainer = ModelTrainer()
        report = trainer.initiate_model_trainer(train_arr, test_arr)

        print("\n Model training complete!")
        print("Final Model Report:")
        for model, metrics in report.items():
            print(f"{model}: R² = {metrics['R²']}, Adjusted R² = {metrics['Adjusted R²']}, MAE = {metrics['MAE']}")

    except Exception as e:
        raise CustomException(e, sys)



# python src/pipeline/train_pipeline.py
