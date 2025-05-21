import os
import sys
os.environ["TMPDIR"] = "C:\\Users\\naren\\AppData\\Local\\Temp"

import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import r2_score

from src.utils import save_object, evaluate_models  # ✅ Custom evaluator updated to return trained models
from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "best_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting data into X/y for train/test")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "SVR": SVR(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(),
                "LightGBM": LGBMRegressor(),
                #"CatBoost": CatBoostRegressor(verbose=False)
            }

            params = {
                "Ridge Regression": {"alpha": [0.1, 1.0, 10.0]},
                "Lasso Regression": {"alpha": [0.01, 0.1, 1.0]},
                "Random Forest": {"n_estimators": [50, 100], "max_depth": [5, 10]},
                "Decision Tree": {"max_depth": [3, 5, 10]},
                "SVR": {"C": [0.1, 1, 10]},
                "Gradient Boosting": {"learning_rate": [0.01, 0.1], "n_estimators": [50, 100]},
                "XGBoost": {"learning_rate": [0.01, 0.1], "n_estimators": [50, 100]},
                "LightGBM": {"learning_rate": [0.01, 0.1], "n_estimators": [50, 100]},
                #"CatBoost": {"learning_rate": [0.01, 0.1], "depth": [4, 6]}
            }

            logging.info("Evaluating models")

            # 🔴 Get both model_report and trained_models from evaluate_models
            model_report, trained_models = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            # 🔴 Find best model based on highest R²
            best_model_name = max(model_report, key=lambda x: model_report[x]["R²"])
            best_model_score = model_report[best_model_name]["R²"]
            best_model = trained_models[best_model_name]  # 🔴 Use trained model here

            if best_model_score < 0.6:
                raise CustomException("No best model found with R² > 0.6")

            logging.info(f"Best model: {best_model_name} with R²: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return model_report

        except Exception as e:
            raise CustomException(e, sys)


# ✅ Test block
if __name__ == "__main__":
    try:
        train_arr = np.load("artifacts/train_arr.npy")
        test_arr = np.load("artifacts/test_arr.npy")

        trainer = ModelTrainer()
        report = trainer.initiate_model_trainer(train_arr, test_arr)

        print("\n Final Model Report:")
        for model_name, scores in report.items():
            print(f"{model_name}: R² = {scores['R²']}, Adjusted R² = {scores['Adjusted R²']}, MAE = {scores['MAE']}")

    except Exception as e:
        print(f"Error: {e}")




#####CODES######################
#pip install numpy pandas

# pip install xgboost catboost lightgbm
