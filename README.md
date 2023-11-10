# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the Credit Card Fraud Detection dataset
# Replace 'creditcard.csv' with the path to your dataset
df = pd.read_csv('creditcard.csv')

# Explore the dataset and perform necessary preprocessing
# For example, check for missing values, handle imbalanced classes, etc.

# Separate features and labels
X = df.drop('Class', axis=1)  # Features
y = df['Class']  # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)
logistic_predictions = logistic_model.predict(X_test_scaled)

# Random Forest
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train_scaled, y_train)
random_forest_predictions = random_forest_model.predict(X_test_scaled)

# Evaluate models
def evaluate_model(predictions, y_true, model_name):
    accuracy = accuracy_score(y_true, predictions)
    confusion_mat = confusion_matrix(y_true, predictions)
    report = classification_report(y_true, predictions)
    
    print(f'{model_name} Model:')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Confusion Matrix:\n{confusion_mat}')
    print('\nClassification Report:\n', report)

# Evaluate Logistic Regression Model
evaluate_model(logistic_predictions, y_test, 'Logistic Regression')

# Evaluate Random Forest Model
evaluate_model(random_forest_predictions, y_test, 'Random Forest')
