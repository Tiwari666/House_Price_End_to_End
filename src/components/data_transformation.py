import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                "amount_requested", "amount_funded_by_investors", "loan_length",
                "debt_to_income_ratio", "monthly_income", "fico_range",
                "open_credit_lines", "revolving_credit_balance", "inquiries_in_the_last_6_months"
            ]

            categorical_columns = [
                "home_ownership", "loan_purpose_grouped",
                "state_region", "employment_length_group"
            ]

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop='first', handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            for df in [train_df, test_df]:
                df.columns = df.columns.str.strip().str.lower().str.replace('.', '_')
                df.replace('.', np.nan, inplace=True)

                df['loan_length'] = df['loan_length'].astype(str).str.extract(r'(\d+)')[0].astype(float)
                df['debt_to_income_ratio'] = df['debt_to_income_ratio'].astype(str).str.replace('%', '').astype(float)
                df['interest_rate'] = df['interest_rate'].astype(str).str.replace('%', '').astype(float)

                df['fico_range'] = df['fico_range'].apply(
                    lambda x: (int(x.split('-')[0]) + int(x.split('-')[1])) / 2 if isinstance(x, str) and '-' in x else np.nan
                )

                emp_map = {
                    '< 1 year': 'very_short', '1 year': 'very_short', '2 years': 'short', '3 years': 'short',
                    '4 years': 'medium', '5 years': 'medium', '6 years': 'medium', '7 years': 'medium',
                    '8 years': 'long', '9 years': 'long', '10+ years': 'long'
                }
                df['employment_length_group'] = df['employment_length'].map(emp_map)

                df['loan_purpose_grouped'] = df['loan_purpose'].replace({
                    'debt_consolidation': 'debt', 'credit_card': 'credit', 'home_improvement': 'home',
                    'major_purchase': 'other', 'small_business': 'other', 'educational': 'other',
                    'vacation': 'other', 'moving': 'other', 'medical': 'other', 'house': 'home',
                    'car': 'other', 'renewable_energy': 'other', 'other': 'other'
                })

                northeast = ['CT','ME','MA','NH','RI','VT','NJ','NY','PA']
                midwest = ['IL','IN','MI','OH','WI','IA','KS','MN','MO','NE','ND','SD']
                south = ['DE','FL','GA','MD','NC','SC','VA','DC','WV','AL','KY','MS','TN','AR','LA','OK','TX']
                west = ['AZ','CO','ID','MT','NV','NM','UT','WY','AK','CA','HI','OR','WA']

                def map_region(state):
                    if state in northeast: return 'northeast'
                    elif state in midwest: return 'midwest'
                    elif state in south: return 'south'
                    elif state in west: return 'west'
                    else: return 'other'

                df['state_region'] = df['state'].apply(map_region)

                cap_cols = [
                'amount_requested', 'amount_funded_by_investors',
                'monthly_income', 'revolving_credit_balance',
                'open_credit_lines', 'interest_rate'
            ]

            for col in cap_cols:
                # 🔴 Ensure numeric conversion to avoid string-based math errors
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower, upper)

            df['inquiries_in_the_last_6_months'] = pd.to_numeric(df['inquiries_in_the_last_6_months'], errors='coerce')


            df['employment_length_group'] = df['employment_length_group'].map({'very_short': 0, 'short': 1, 'medium': 2, 'long': 3})
            df['loan_length'] = df['loan_length'].map({36.0: 0, 60.0: 1})

            target_column = 'interest_rate'
            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]
            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            preprocessing_obj = self.get_data_transformer_object()
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_df.values.reshape(-1, 1)]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df.values.reshape(-1, 1)]

            # 🔽 Save transformed arrays to disk
            np.save(os.path.join('artifacts', 'train_arr.npy'), train_arr)
            np.save(os.path.join('artifacts', 'test_arr.npy'), test_arr)


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    transformer = DataTransformation()
    train_arr, test_arr, path = transformer.initiate_data_transformation(
        "artifacts/train.csv", "artifacts/test.csv"
    )
    print("Transformed Train Shape:", train_arr.shape)
    print("Transformed Test Shape:", test_arr.shape)


# python src/components/data_transformation.py