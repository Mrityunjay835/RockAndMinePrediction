# Sonar Rock vs Mine Classification using Logistic Regression

## Overview
This project implements a machine learning model using Logistic Regression to classify sonar signals as either rocks (R) or mines (M). The dataset used for this project is the Sonar dataset.

## Dataset
- The dataset is loaded from a CSV file (`sonar data.csv`).
- It contains 208 rows and 60 numerical features representing sonar signal strengths at different frequencies.
- The last column (label) indicates whether the object detected is a **Mine (M)** or a **Rock (R)**.

## Installation
Ensure you have Python installed along with the required dependencies. You can install the necessary libraries using:

```sh
pip install numpy pandas scikit-learn
```

## Steps
1. **Importing Dependencies**
   - NumPy, Pandas, scikit-learn (for machine learning model)

2. **Loading and Exploring Data**
   - Load dataset into a Pandas DataFrame
   - Display summary statistics
   - Count occurrences of each label (R/M)
   
3. **Data Preprocessing**
   - Splitting features (`X`) and labels (`Y`)
   - Train-test split (90% training, 10% testing, stratified by label)

4. **Model Training**
   - Logistic Regression model is used
   - Model is trained on the training data

5. **Model Evaluation**
   - Training accuracy: ~83%
   - Test accuracy: ~76%

6. **Prediction System**
   - Takes new sonar data as input
   - Predicts if the object is a Mine (M) or Rock (R)

## Usage
To run the model and make predictions:

```python
# Load and train the model
python sonar_classification.py
```

To test with new data:
```python
input_data = (0.0162,0.0041,0.0239,0.0441,0.0630,...) # Example input
prediction = model.predict(np.asarray(input_data).reshape(1, -1))
print("Predicted label:", prediction)
```

## Future Improvements
- Try different ML models (SVM, Random Forest, Neural Networks)
- Improve accuracy using feature selection and hyperparameter tuning
- Deploy as a web app using Flask or FastAPI

## License
This project is open-source and available under the MIT License.


