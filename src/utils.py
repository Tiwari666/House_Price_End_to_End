import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}  # ✅ model report dict
        trained_models = {}  # 🔴 Added to store trained versions

        for model_name, model in models.items():
            print(f"\n🔍 Tuning and Evaluating: {model_name}")

            if param.get(model_name):
                gs = GridSearchCV(model, param[model_name], cv=3, scoring='r2', n_jobs=-1)
                gs.fit(X_train, y_train)
                model = gs.best_estimator_  # 🔴 Pick trained best estimator
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            n = X_test.shape[0]
            p = X_test.shape[1]
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            mae = mean_absolute_error(y_test, y_pred)

            report[model_name] = {
                "R²": round(r2, 4),
                "Adjusted R²": round(adj_r2, 4),
                "MAE": round(mae, 4),
            }

            trained_models[model_name] = model  # 🔴 Store trained model

        return report, trained_models  # 🔴 Return both

    except Exception as e:
        raise CustomException(e, sys)


# ✅ Optional test block
if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    dummy_path = os.path.join("artifacts", "dummy_model.pkl")

    save_object(dummy_path, model)
    loaded_model = load_object(dummy_path)

    print(" Model saved and loaded successfully:", type(loaded_model))



# #########code to run the utils.py module
#Do not forget to add this part on the top before you run the code:
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# #########code to run the utils.py module
# python src/utils.py
