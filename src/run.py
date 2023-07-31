"""
This module outputs the performance of the model on slices of the data for categorical features.
"""
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from model_utils import load_artifact, process_data
from model import compute_model_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

DATA_PATH = 'data/clean_census.csv'
ARTIFACTS_PATH = 'model'

# Categorical features
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

def go():
    """ Check performance on categorical features """

    data = pd.read_csv(DATA_PATH)
    _, test = train_test_split(data, test_size=0.20)

    rf = load_artifact(os.path.join(ARTIFACTS_PATH, 'model.pkl'))
    encoder = load_artifact(os.path.join(ARTIFACTS_PATH, 'encoder.pkl'))
    lb = load_artifact(os.path.join(ARTIFACTS_PATH, 'lb.pkl'))

    slice_metrics = []

    for feature in cat_features:
        for cls in test[feature].unique():
            df_temp = test[test[feature] == cls]

            X_test, y_test, _, _ = process_data(
                df_temp,
                cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                training=False)

            y_pred = rf.predict(X_test)

            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
            row = f"{feature} - {cls} :: Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}"
            slice_metrics.append(row)

            with open('model/metrics.txt', 'w') as file:
                for row in slice_metrics:
                    file.write(row + '\n')

    logging.info("Performance metrics for slices saved to slice_output.txt")


if __name__ == '__main__':
    go()