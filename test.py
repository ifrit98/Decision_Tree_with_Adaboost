from FeatureExtraction import FeatureParser
from ada import WeightedMajority
import pickle

def predict(hypIn, test, learning_type):
    """
    Given a pickled hypothesis model, test it on new data and print predictions to stout.
    :param hypIn: Hypothesis filename
    :param test: Test data filename
    :param learning_type: Adaboost or Decision Tree type
    :return: None
    """
    # Parse features from test data
    feat_obj = FeatureParser(test, True)

    # Unpickle model
    hypothesis_file = open(hypIn, 'rb')
    model = pickle.load(hypothesis_file)

    # Make predictions via tree traversal and print to stout
    predictions = []
    if learning_type == 'dt':
        for example in feat_obj.features:
            pred = model(example)
            predictions.append(pred)
            print(pred)
    else: # If Adaboost
        hypotheses, z = model[0], model[1]
        predict = WeightedMajority(hypotheses, z)
        for example in feat_obj.features:
            pred = predict(example)
            predictions.append(pred)
            print(pred)

    return predictions

def evaluate(predictions, label_fn):
    assert label_fn, 'Must provide list of correct labels for test data to evaluate.'
    labels = []
    with open(label_fn, 'r') as f:
        for label in f:
            labels.append(label.strip())

    assert len(labels) == len(predictions), 'Number of test examples must match number of labels'

    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            correct += 1

    return correct / len(predictions) * 100