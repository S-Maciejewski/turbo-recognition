import os

from model import Car
from model_training import train_model
from predictions import make_prediction

if __name__ == '__main__':
    if not os.path.exists('model.pkl'):
        train_model()

    car = Car(
        id=1,
        name='Cupra Formentor VZ',
        engine_size=2.0,
        cylinders=4,
        turbo=None,
        petrol=1,
        combined_fuel_consumption=7.7
    )

    make_prediction(car)
