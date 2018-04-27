from FeatureExtraction import FeatureData, FeatureParser
from train import train
from test import predict, evaluate
from sys import argv

def main():
    """
    Main driver program for lab 2.  Takes in cmd-line args,
    creates a feature object, parses the training examples
    and creates a data-set object that will be fed into the
    Decision Tree and Adaboost algorithms for training from argv[1].
    train() then pickles (serializes) the model returned from training
    and predict() unpickles the model and makes predictions based on
    test data input via argv[4]
    :return: None
    """

    # Command-line arguments
    training_data = argv[1]
    hypothesis_out = argv[2]
    learning_type = argv[3]
    test = argv[4]
    labels = None
    if len(argv) > 5:
        labels = argv[5]

    # Parse data and determine features
    feat_obj = FeatureParser(training_data)
    data = FeatureData(feat_obj.features)

    # Train model using DT or DT + adaboost
    train(data, hypothesis_out, learning_type)

    # Predict on test set with trained model
    predictions = predict(hypothesis_out, test, learning_type)

    # Evaluate accuracy of test data if provided lables
    if labels:
        accuracy = evaluate(predictions, labels)
        print('Model accuracy on test data:',str(accuracy) + '%')

if __name__ == '__main__':
    main()