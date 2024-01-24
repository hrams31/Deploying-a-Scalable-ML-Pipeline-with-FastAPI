import pandas as pd
import numpy as np
from ml.data import process_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from ml.model import train_model
from ml.model import compute_model_metrics
# from ml.model import inference

from sklearn.preprocessing import OneHotEncoder, LabelBinarizer


# TODO: implement the first test. Change the function name and input as needed

def data_input():
    """Obtain data for testing

    Returns:

    pd.DataFrame -- holds the testing data
    """

    X, y = make_classification(n_samples=1500, n_features=5, random_state=1089)
    input_data = pd.DataFrame(X, columns=['feature_zero', 'feature_one',
                                          'feature_two', 'feature_three',
                                          'feature_four'])
    input_data['feature_one'] = pd.qcut(input_data['feature_one'], 5,
                                        labels=['a', 'b', 'c', 'd', 'e'])
    input_data['feature_three'] = pd.qcut(input_data['feature_three'], 5,
                                          labels=['v', 'w', 'x', 'y', 'z'])
    input_data['label'] = y
    return input_data

# test that the trained model is a random forest classifier

def test_train_model():


    """Tests to make sure the model is a random classifier model;
    predict attribute is not none."""

    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 0])

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, 'predict')

    assert model is not None

# TODO: implement the second test. Change the function name and input as needed


def test_compute_model_metrics():

    # labels
    y_true = np.array([1, 0, 1, 1, 0, 1])
    preds = np.array([1, 0, 1, 0, 0, 1])

    # computing precision, recall fbeta
    precision, recall, fbeta = compute_model_metrics(y_true, preds)

    # make sure all metrics are correct type - float
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


# TODO: implement the third test. Change the function name and input as needed
def test_process_data_training_mode():

    # get dummy training data
    data = {
        'feature_one': ['a', 'b', 'c', 'a'],
        'feature_two': [1, 2, 3, 4],
        'label': [0, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    # Call the process_data function in training mode
    X, y, encoder, lb = process_data(df, categorical_features=['feature_one'],
                                     label='label', training=True)

    # Obtain assertions to verify the processed data
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(lb, LabelBinarizer)
    assert isinstance(encoder, OneHotEncoder)
