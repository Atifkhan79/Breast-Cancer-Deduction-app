🩺 Breast Cancer Diagnostic Predictor

A machine learning web application that predicts whether a breast mass is benign or malignant from 30 digitized cell-nucleus measurements, based on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. Built with a Logistic Regression model wrapped in an sklearn.Pipeline (with feature scaling), served through Flask and Gunicorn, and deployed as a Docker container.

🔗 Live demo: https://breast-cancer-deduction-app--atif79918.replit.app/

⚠️ Disclaimer: This is an educational/portfolio project only. It is not a certified medical device and must not be used for real diagnostic decisions.

📌 Project Overview

Breast cancer diagnosis traditionally relies on a pathologist manually examining a Fine Needle Aspirate (FNA) sample of a breast mass. This project demonstrates how machine learning can assist that process by learning patterns from 30 quantitative measurements of cell nuclei (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension — each reported as a mean, standard error, and "worst" value) and predicting whether the mass is benign or malignant, along with a confidence score.

✨ Key Features
End-to-end ML pipeline — StandardScaler + LogisticRegression bundled together in a single sklearn.Pipeline, so raw feature values go in and a correctly scaled prediction comes out (no manual preprocessing required at inference time).
Clean, guided web form — Inputs are organized into the same mean / standard error / worst structure as the original dataset, grouped by measurement, instead of a single box requiring comma-separated values.
Pre-filled sample data — The form loads with a real sample from the dataset so first-time visitors immediately see the tool working.
Confidence score — Every prediction is returned alongside a probability/confidence percentage, with a visual flag for low-confidence results.
Input validation — Submissions are checked for valid numeric input before scoring, with clear feedback on which fields need correction.
Production-ready serving — Runs behind Gunicorn inside a lightweight Docker container, ready to deploy on any container-based hosting platform.
🧠 Model Details
	
Algorithm	Logistic Regression
Preprocessing	StandardScaler (fitted and saved as part of the pipeline)
Target	Binary classification — benign (0) vs malignant (1)
Output	Predicted class + confidence percentage

The trained pipeline and the exact feature ordering used during training are serialized together in model.pkl as a single bundle, which removes the risk of a scaler/model mismatch at inference time.

🛠️ Tech Stack
Backend: Python, Flask
ML: scikit-learn, NumPy
Serving: Gunicorn
Containerization: Docker
Frontend: HTML/CSS (Jinja2 templates)
📂 Project Structure
.
├── app.py              # Flask application (routes, form handling, prediction)
├── model.pkl            # Serialized {"pipeline": ..., "feature_names": [...]}
├── requirements.txt     # Python dependencies
├── Dockerfile            # Container build definition
├── templates/
│   └── index.html       # Web form + results page
└── static/
    └── style.css         # Styling
	
📊 How It Works
The user fills in 30 measurements via the web form (or uses the pre-filled sample).
On submit, app.py validates each field and assembles the values into the exact feature order the model was trained on.
The sklearn.Pipeline scales the input and runs it through the logistic regression model.
The predicted class and confidence score are rendered back to the user.
📈 Possible Future Improvements
Add model explainability (e.g. SHAP values) to show which measurements most influenced a given prediction.
Expand to compare multiple model types (Random Forest, SVM, XGBoost).
Add automated tests and CI/CD for deployment.
Add authentication/logging for real-world usage tracking.
👤 Author

Atif Khan 🔗 Portfolio: atifkhan7portfolio.netlify.app
