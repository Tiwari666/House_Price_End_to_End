# src/pipeline/predict_pipeline.py

import os, sys
os.environ["TMPDIR"] = "C:\\Users\\naren\\AppData\\Local\\Temp"
import numpy as np
import pandas as pd
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "best_model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features):
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            return prediction

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
        self,
        amount_requested: float,
        amount_funded_by_investors: float,
        loan_length: float,
        debt_to_income_ratio: float,
        monthly_income: float,
        fico_range: float,
        open_credit_lines: float,
        revolving_credit_balance: float,
        inquiries_in_the_last_6_months: float,
        home_ownership: str,
        loan_purpose_grouped: str,
        state_region: str,
        employment_length_group: int
    ):
        self.amount_requested = amount_requested
        self.amount_funded_by_investors = amount_funded_by_investors
        self.loan_length = loan_length
        self.debt_to_income_ratio = debt_to_income_ratio
        self.monthly_income = monthly_income
        self.fico_range = fico_range
        self.open_credit_lines = open_credit_lines
        self.revolving_credit_balance = revolving_credit_balance
        self.inquiries_in_the_last_6_months = inquiries_in_the_last_6_months
        self.home_ownership = home_ownership
        self.loan_purpose_grouped = loan_purpose_grouped
        self.state_region = state_region
        self.employment_length_group = employment_length_group

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                "amount_requested": [self.amount_requested],
                "amount_funded_by_investors": [self.amount_funded_by_investors],
                "loan_length": [self.loan_length],
                "debt_to_income_ratio": [self.debt_to_income_ratio],
                "monthly_income": [self.monthly_income],
                "fico_range": [self.fico_range],
                "open_credit_lines": [self.open_credit_lines],
                "revolving_credit_balance": [self.revolving_credit_balance],
                "inquiries_in_the_last_6_months": [self.inquiries_in_the_last_6_months],
                "home_ownership": [self.home_ownership],
                "loan_purpose_grouped": [self.loan_purpose_grouped],
                "state_region": [self.state_region],
                "employment_length_group": [self.employment_length_group]
            }
            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e, sys)

# test code for prediction
def predict(sample_input):
    try:
        pipeline = PredictPipeline()
        return pipeline.predict(sample_input)
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    sample = CustomData(
        amount_requested=12000,
        amount_funded_by_investors=12000,
        loan_length=0,
        debt_to_income_ratio=15.0,
        monthly_income=5000,
        fico_range=700,
        open_credit_lines=8,
        revolving_credit_balance=4500,
        inquiries_in_the_last_6_months=1,
        home_ownership="MORTGAGE",
        loan_purpose_grouped="debt_consolidation",
        state_region="West",
        employment_length_group=2
    )

    sample_input = sample.get_data_as_dataframe()
    result = predict(sample_input)
    print(f"\n Predicted Interest Rate: {round(result[0], 2)}%")





# manually delete the best_model.pkl, if problem arises.
# Re-run the training pipeline to re-save a valid model

# python src/pipeline/train_pipeline.py



# python src/pipeline/predict_pipeline.py 