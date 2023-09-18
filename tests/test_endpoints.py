""" This module tests the root and the prediction end points """
from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)


def test_get_root():
    """ Test the root page get a succesful response"""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "Hi": "This app predicts wether income exceeds $50K/yr based on census data."}


def test_post_predict_up():
    """ Test an example when income is less than 50K """

    r = client.post("/predict-income", json={
        "age": 37,
        "workclass": "Private",
        "fnlgt": 280464,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 80,
        "native_country": "United-States"
    })

    assert r.status_code == 200
    assert r.json() == {"Income prediction": ">50K"}


def test_post_predict_down():
    """ Test an example when income is higher than 50K """
    r = client.post("/predict-income", json={
        "age": 28,
        "workclass": "Private",
        "fnlgt": 183175,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Divorced",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })

    assert r.status_code == 200
    assert r.json() == {"Income prediction": "<=50K"}