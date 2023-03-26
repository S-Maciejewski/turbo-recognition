import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle


def train_model():
    # Load the dataset
    data = pd.read_csv('./data/car_data_clean.csv')

    # Select the required columns
    x = data[['engine_size', 'cylinders', 'petrol', 'combined_fuel_consumption']]
    y = data['turbo']

    # Split the dataset into training and test sets (80% training, 20% test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Train the logistic regression model
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Save the trained model and scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump((model, scaler), f)
