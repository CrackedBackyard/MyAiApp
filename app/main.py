from fastapi import FastAPI
import torch
import numpy as np

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, world!"}

@app.get("/predict")
def predict(input_data: str):
    # Example model prediction logic (replace with your actual model)
    model = torch.load('model.pth')
    prediction = model(torch.tensor([float(input_data)]))
    return {"prediction": prediction.item()}
