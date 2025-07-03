import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib  # untuk save/load model dan scaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    cols_with_zero_invalid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zero_invalid] = df[cols_with_zero_invalid].replace(0, pd.NA)
    df = df.dropna()
    return df

def split_and_scale(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    data_path = 'diabetes.csv'
    df = load_data(data_path)
    df = preprocess_data(df)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(df)

    # Pilih model yang ingin dipakai:
    model = train_random_forest(X_train_scaled, y_train)
    # model = train_logistic_regression(X_train_scaled, y_train)

    evaluate_model(model, X_test_scaled, y_test)

    # Save model dan scaler (opsional)
    joblib.dump(model, 'random_forest_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
