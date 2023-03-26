import pandas as pd
import pickle

from model import Car


def make_prediction(car: Car):
    # Load the trained model and scaler
    with open('model.pkl', 'rb') as f:
        model, scaler = pickle.load(f)

    # Convert the Car object to a DataFrame, keeping only the required columns for prediction
    new_data = pd.DataFrame([{
        'engine_size': car.engine_size,
        'cylinders': car.cylinders,
        'petrol': car.petrol,
        'combined_fuel_consumption': car.combined_fuel_consumption
    }])

    # Scale the features
    new_data_scaled = scaler.transform(new_data)

    # Make the prediction
    prediction = model.predict(new_data_scaled)[0]

    # Print the prediction result
    if prediction == 1:
        print("The car is predicted to be turbocharged.")
    else:
        print("The car is predicted to be naturally aspirated.")
