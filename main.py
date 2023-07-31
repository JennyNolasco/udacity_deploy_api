"""
This module contains the code for the API
"""
import os
from fastapi import FastAPI
from typing import Literal
from pandas import DataFrame
import numpy as np
import uvicorn
from pydantic import BaseModel
from src.model_utils import load_artifact, process_data, get_cat_features
from src.model import inference

# Set up DVC on Heroku
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    print("Running DVC")
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("Pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")



# Instantiate the app.
app = FastAPI()

class Model(BaseModel):
    age: int

    workclass: Literal['Private', 'Self-emp-inc', 'Self-emp-not-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay']

    fnlgt: int

    education: Literal['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Masters', 'Doctorate', 'Prof-school']

    education_num: int

    marital_status: Literal["Married-AF-spouse", "Married-civ-spouse", "Married-spouse-absent", "Never-married", "Separated", "Divorced", "Widowed"]
    
    occupation: Literal["Armed-Forces", "Craft-repair", "Exec-managerial", "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct", "Other-service", "Priv-house-serv", "Prof-specialty", "Protective-serv", "Sales", "Tech-support", "Transport-moving", "Adm-clerical"]
    
    relationship: Literal["Husband", "Not-in-family", "Other-relative", "Own-child", "Unmarried", "Wife"]
    
    race: Literal["Asian-Pac-Islander", "Amer-Indian-Eskimo", "Black", "Other", "White"]
    
    sex: Literal["Female", "Male"]
    
    capital_gain: int
    
    capital_loss: int
    
    hours_per_week: int
    
    native_country: Literal['Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia']

    class Config:
        schema_extra = {
            "example": {
                "age": 34,
                "workclass": 'Self-emp-inc',
                "fnlgt": 77516,
                "education": 'Masters',
                "education_num": 14,
                "marital_status": "Married-civ-spouse",
                "occupation": "Other-service",
                "relationship": "Wife",
                "race": "White",
                "sex": "Female",
                "capital_gain": 5000,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": 'Portugal'
            }
        }


# Load artifacts
model = load_artifact("model/model.pkl")
encoder = load_artifact("model/encoder.pkl")
lb = load_artifact("model/lb.pkl")


# Root path
@app.get("/")
async def root():
    return {
        "Message": "Welcome"}

# Prediction path
@app.post("/inference")
async def predict(input: Model):

    input_data = np.array([[
        input.age,
        input.workclass,
        input.fnlgt,
        input.education,
        input.education_num,
        input.marital_status,
        input.occupation,
        input.relationship,
        input.race,
        input.sex,
        input.capital_gain,
        input.capital_loss,
        input.hours_per_week,
        input.native_country]])

    original_cols = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours-per-week",
        "native-country"]

    input_df = DataFrame(data=input_data, columns=original_cols)
    cat_features = get_cat_features()

    X, _, _, _ = process_data(
        input_df, categorical_features=cat_features, encoder=encoder, lb=lb, training=False)
    y = inference(model, X)
    pred = lb.inverse_transform(y)[0]

    return {"Income prediction": pred}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
