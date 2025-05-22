## Linear Regression Bank Loan

## Linear Regression Bank Loan: Loan Interest Rate Prediction Web App

A full end-to-end machine learning project that predicts **loan interest rates** based on applicant data. Built using **Python**, **Scikit-learn**, and deployed via **Flask**, **Docker**, and **Render Cloud**.

---

##  Key Features

* **Data Preprocessing**: Cleaned raw loan data (e.g., missing values, percentage parsing, category grouping)
* **Model Training**: Compared multiple regressors including Linear, Ridge, Lasso, Random Forest, XGBoost, LightGBM, and CatBoost
* **Best Model Selection**: Saved top-performing model based on R² and MAE
* **Web Interface**: Flask-based form for user input and real-time prediction
* **Dockerized**: Built Docker image for consistent deployment
* **Cloud Deployed**: Hosted using Render for global access

---

##  Project Structure

```bash
LR_BankLoan/
├── artifacts/                # Saved models (best_model.pkl, preprocessor.pkl)
├── src/                     # Modular ML pipeline (data ingestion, transformation, training)
│   ├── components/
│   ├── pipeline/
│   ├── utils.py
├── FlaskApp/                # Flask app
│   ├── app.py
│   ├── templates/home.html
│   └── static/
├── notebook/                # EDA, modeling, evaluation notebooks
├── Dockerfile               # Docker build instructions
├── .dockerignore            # Avoid unnecessary files in Docker
├── render.yaml              # Render deployment configuration
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md                # You are here
```

---

## Technologies Used

| Task             | Tools/Tech                           |
| ---------------- | ------------------------------------ |
| EDA & Modeling   | Python, Pandas, Matplotlib, Seaborn  |
| ML Algorithms    | Sklearn, XGBoost, LightGBM, CatBoost |
| Web Deployment   | Flask                                |
| Containerization | Docker                               |
| Cloud Hosting    | Render Cloud                         |

---

## UI Preview

> Real-time prediction form hosted on Render: accepts 13 loan features and returns interest rate prediction

---

## How to Run Locally

```bash
# Clone the repository
https://github.com/Tiwari666/LR_BANKLOAN.git

# Navigate to project folder
cd LR_BANKLOAN

# Install dependencies
pip install -r requirements.txt

# Run Flask app
cd FlaskApp
python app.py

# Visit
http://127.0.0.1:5000/
```

---

## Docker Deployment

```bash
# Build Docker image
docker build -t loan-predictor-app .

# Run Docker container
docker run -p 5000:5000 loan-predictor-app
```

---

## Cloud Deployment via Render

* Pushed project to GitHub
* Added `render.yaml`
* Connected repo with Render
* Auto-deployed using Dockerfile setup
* Public URL: \[Your Render App Link Here]

---

##  Sample Input

```
amount_requested: 12000
amount_funded_by_investors: 11500
loan_length: 36
debt_to_income_ratio: 0.18
monthly_income: 4200
fico_range: 710
open_credit_lines: 7
revolving_credit_balance: 3200
inquiries_in_the_last_6_months: 1
home_ownership: RENT
loan_purpose_grouped: DEBTCONSOLIDATION
state_region: WEST
employment_length_group: 2
```

---

##  Best Performing Model

```
CatBoost Regressor:
R² = 0.6829
Adjusted R² = 0.6646
MAE = 1.708
```

---

##  Author

**Narendra Tiwari**
Data Science Enthusiast
[GitHub Profile](https://github.com/Tiwari666)

---

## Contributions & Feedback

Have suggestions or improvements? Feel free to raise issues or pull requests.

>  Thanks for checking out the project!

 
