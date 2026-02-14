import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

DATASET_FILE = "weather_data.csv"
MODEL_FILE = "weather_model.pkl"

def train_model():
    print("Training model...")

    df = pd.read_csv(DATASET_FILE)

    X = df[['temperature', 'humidity', 'pressure']]

    y_temp = df['tomorrow_temperature']
    y_rain = df['rainy']

    X_train, X_test, y_temp_train, y_temp_test = train_test_split(
        X, y_temp, test_size=0.2, random_state=42
    )

    _, _, y_rain_train, y_rain_test = train_test_split(
        X, y_rain, test_size=0.2, random_state=42
    )

    temp_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rain_model = RandomForestClassifier(n_estimators=200, random_state=42)

    temp_model.fit(X_train, y_temp_train)
    rain_model.fit(X_train, y_rain_train)

    joblib.dump((temp_model, rain_model), MODEL_FILE)

    print("Models saved successfully!")

if __name__ == "__main__":
    train_model()
