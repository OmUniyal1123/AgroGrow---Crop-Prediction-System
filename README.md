AgroGrow - Crop Prediction System 🌱
AgroGrow is a machine learning-based system designed to assist farmers and agricultural enthusiasts in identifying the most suitable crop to grow based on soil and environmental conditions. By leveraging data-driven insights, AgroGrow empowers users to make informed decisions, improving productivity and sustainability.

**Features 🚀**
Accurate Crop Recommendations: Predicts the most suitable crop based on key environmental parameters.

Interactive Web Interface: Dynamically input data via a user-friendly website.

Model Comparison and Selection: Evaluates multiple machine learning models to ensure the best accuracy.

**Model Comparison and Selection 📊**
During the development of AgroGrow, three machine learning models were evaluated to determine the best fit for crop prediction based on their accuracy scores:

Logistic Regression: 94.55%

Support Vector Machine (SVC): 96.14%

Random Forest: 99.32%

After analyzing the results, Random Forest was chosen as the final model for its superior accuracy and ability to handle complex patterns in the data effectively.

**Dataset 📊**
This project utilizes the Crop Recommendation Dataset from Kaggle. The dataset includes the following features:

N, P, K: Soil macronutrients (Nitrogen, Phosphorus, Potassium).

Temperature: Average temperature (°C).

Humidity: Percentage of humidity.

pH: Acidity or alkalinity of the soil.

Rainfall: Annual rainfall (mm).

Label: Crop type.

🔗 Crop Recommendation Dataset on Kaggle
**
Technologies Used 🛠️**
Frontend: HTML, CSS

Backend: Flask, Jupyter Notebook

Libraries: Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)

**How It Works 🤖**
Data Input: Users input soil and environmental parameters (N, P, K, temperature, humidity, pH, rainfall) via the web interface.

Model Prediction: The input data is fed into the trained machine learning model.

Output: The system predicts the most suitable crop to grow based on the input parameters.

**Acknowledgments 🙏**
Kaggle for providing the dataset.

Scikit-learn Documentation for machine learning resources.

**Contact 📧**
For any inquiries or feedback, feel free to reach out:
📩 Email: omuniyal0@gmail.com


