
# Loan Interest Rate Prediction App

A full-stack machine learning project that predicts interest rates for bank loans based on various applicant features. This project includes data preprocessing, model training, Flask-based web interface, Dockerization, and cloud deployment using Render.

--------
##  Key Features

* **Data Preprocessing**: Cleaned raw loan data (e.g., missing values, percentage parsing, category grouping)
* **Model Training**: Compared multiple regressors including Linear, Ridge, Lasso, Random Forest, XGBoost, LightGBM, and CatBoost
* **Best Model Selection**: Saved top-performing model based on R² and MAE
* **Web Interface**: Flask-based form for user input and real-time prediction
* **Dockerized**: Built Docker image for consistent deployment
* **Cloud Deployed**: Hosted using Render for global access

--------


---

##  Machine Learning Pipeline

* **Model Used**: Multiple regression models (Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost)
* **Best Model**: CatBoost (R² = 0.6829)
* **Features Engineered**:

  * Employment Length Group
  * State Region
  * Loan Purpose Grouped
  * Debt to Income Ratio
  * FICO Range

---

##  Project Structure

```bash
LR_BankLoan/
├── artifacts/                  # Saved model + preprocessor
│   ├── best_model.pkl
│   └── preprocessor.pkl
├── FlaskApp/                  # Flask Web App
│   ├── app.py                 # Main Flask file
│   ├── templates/
│   │   └── home.html
│   └── static/
├── notebook/                  # Jupyter notebooks (EDA, Modeling, etc.)
├── src/                       # Source ML pipeline modules
│   ├── components/
│   ├── pipeline/
│   ├── utils.py
│   └── exception.py
├── Dockerfile                 # For containerization
├── .dockerignore              # Ignore unneeded files in Docker
├── requirements.txt           # Python dependencies
├── render.yaml                # For Render cloud deployment
└── README.md
```

---

##  ML Pipeline Modules

| Module                | Function/Class                  | Purpose                            |
| --------------------- | ------------------------------- | ---------------------------------- |
| `data_ingestion`      | `DataIngestion`                 | Load and split raw data            |
| `data_transformation` | `DataTransformation`            | Preprocess, scale, and encode data |
| `model_trainer`       | `ModelTrainer`                  | Train and evaluate models          |
| `predict_pipeline`    | `CustomData`, `PredictPipeline` | Prepare input and predict          |
| `utils.py`            | `save_object`, `load_object`    | Serialize model + evaluation logic |

---

##  Flask Web Interface

* Collect user inputs via form
* Predicts interest rate using the best trained model
* Displays result directly on the browser

📸 **Flask UI**

![data_entry](https://github.com/user-attachments/assets/6c4988ea-8697-44df-b668-d9c5eb8e9841)
![Result](https://github.com/user-attachments/assets/cf9b507e-16f1-4383-b430-fd8c539db115)

---

## Dockerization

* Dockerfile created to containerize the Flask app
* Built image:

```bash
docker build -t loan-predictor-app .
docker run -p 5000:5000 loan-predictor-app
```

 **Docker Running Screenshot**


![Docker_predicted_interest_rate](https://github.com/user-attachments/assets/b2f03338-c337-4782-94c4-a4162151bafa)
![Docker_data_interest_rate_prediction_combo](https://github.com/user-attachments/assets/8f7bad30-ee9e-4093-ae48-e894a37e6949)



---

## Cloud Deployment via Render

* `render.yaml` used for automated deployment
* Connected GitHub repo to Render
* Live prediction on public URL: [https://dashboard.render.com/web/srv-d0njllmuk2gs73c2aehg/deploys/dep-d0novih5pdvs73b2ejrg](https://lr-bankloan.onrender.com/)

 **Render Deployed App Screenshot**

![Render_form](https://github.com/user-attachments/assets/00d7caca-20e2-4d39-9d97-43234a385355)
![render_deployment](https://github.com/user-attachments/assets/441057fc-e9dd-46e2-b6a7-4c3a35e6f9f7)

---

##  Highlights

*  End-to-End ML lifecycle completed
*  Docker & Cloud deployment ready
*  Model evaluation + hyperparameter tuning
*  Robust data cleaning pipeline

---

##  Requirements

```bash
flask
pandas
numpy
scikit-learn
xgboost
lightgbm
catboost
seaborn
matplotlib
```

---

##  Future Enhancements

* Add more models (e.g., neural networks)
* CI/CD integration with GitHub Actions
* Input validation & UI improvements
* Logging and monitoring

---

##  Author

**Narendra Tiwari**
 [GitHub Profile](https://github.com/Tiwari666)

---

**Thanks for visiting!** 
