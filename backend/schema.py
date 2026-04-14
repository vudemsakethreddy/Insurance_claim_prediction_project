from pydantic import BaseModel

class PredictRequest(BaseModel):
    age: float
    gender: int
    bmi: float
    children: int
    smoker: int
    region: int
    charges: float