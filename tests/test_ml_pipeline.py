"""
This module includes unit tests for the ML model

"""
from pathlib import Path
import logging
import pandas as pd
import pytest
from src.model_utils import load_artifact, process_data, get_cat_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


DATA_PATH = 'data/clean_census.csv'
MODEL_PATH = 'model/model.pkl'

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture(name='data')
def data():
    """
    Fixture will be used by the unit tests.
    """
    yield pd.read_csv(DATA_PATH)


def test_load_data(data):
    
    """ Check the data received """

    assert isinstance(data, pd.DataFrame)
    assert data.shape[0]>0
    assert data.shape[1]>0


def test_model():

    """ Check model type """

    model = load_artifact(MODEL_PATH)
    assert isinstance(model, RandomForestClassifier)


def test_process_data(data):

    """ Test the data split """

    train, _ = train_test_split(data, test_size=0.20)
    X, y, _, _ = process_data(train, cat_features, label='salary')
    assert len(X) == len(y)