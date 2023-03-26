from dataclasses import dataclass


@dataclass
class Car:
    id: int
    name: str
    engine_size: float
    cylinders: int
    turbo: int
    petrol: int
    combined_fuel_consumption: float
