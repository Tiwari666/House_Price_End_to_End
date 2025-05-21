from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home_page():
    if request.method == "GET":
        return render_template("home.html")  
    else:
        # Process the form submission
        try:
            # Extract input values
            data = CustomData(
                amount_requested=float(request.form["amount_requested"]),
                amount_funded_by_investors=float(request.form["amount_funded_by_investors"]),
                loan_length=float(request.form["loan_length"]),
                debt_to_income_ratio=float(request.form["debt_to_income_ratio"]),
                monthly_income=float(request.form["monthly_income"]),
                fico_range=float(request.form["fico_range"]),
                open_credit_lines=float(request.form["open_credit_lines"]),
                revolving_credit_balance=float(request.form["revolving_credit_balance"]),
                inquiries_in_the_last_6_months=float(request.form["inquiries_in_the_last_6_months"]),
                home_ownership=request.form["home_ownership"],
                loan_purpose_grouped=request.form["loan_purpose_grouped"],
                state_region=request.form["state_region"],
                employment_length_group=int(request.form["employment_length_group"])
            )

            # Transform into dataframe and predict
            final_data = data.get_data_as_dataframe()
            pipeline = PredictPipeline()
            prediction = pipeline.predict(final_data)

            return render_template("home.html", result=round(prediction[0], 2))

        except Exception as e:
            return f"Error occurred: {e}"

# Move inside `if __name__ == "__main__"` to avoid running when imported
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # Set debug=True for dev; set False when deploying



# pip install flask

# C:\Users\naren\Desktop\New_DS\LR_BankLoan>

# python FlaskApp/app.py

# To exit the running Flask development server in the terminal, simply Press:

# Ctrl + C

# This will stop the server and return you to the command prompt. 
