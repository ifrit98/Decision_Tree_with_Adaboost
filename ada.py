from collections import defaultdict
from math import log

def adaboost(learner, iterations):
    """
    Initial caller for adaboost.
    :param learner: Input learning algorithm, i.e. decision tree
    :param iterations: Number of weak classifiers (stumps) desired
    :return: the train() function to be called
    """

    def train(data):
        """
        Main adaboost training function.  Create K stumps and update weight values
        for each stump, then combines weak classifiers.
        :param data: Data-set with examples to train on
        :return: A list containing a list of hypotheses (stumps) and
        a list of hypotheses and weights associated with each hypothesis (stump)
        """
        examples, label = data.examples, data.label
        N = len(examples)
        epsilon = 1 / (2 * N)
        weights = [1 / N] * N
        hypotheses, W = [], []
        for i in range(iterations):
            hypothesis_i = learner(data, weights)
            hypotheses.append(hypothesis_i)
            error = sum(weight for example, weight in zip(examples, weights)
                        if example[label] != hypothesis_i(example))

            # Avoid divide by zero error
            error = clip(error, epsilon, 1 - epsilon)
            # Update weights for each example based on the given learner
            for i, example in enumerate(examples):
                if example[label] == hypothesis_i(example):
                    weights[i] *= error / (1 - error)
            weights = normalize(weights)
            W.append(log((1 - error) / error))

        return [hypotheses, W]

    return train


def WeightedMajority(predictors, weights):
    """
    Predicts classification of examples for test data
    :param predictors: Hypotheses (stumps)
    :param weights: Hypothesis weights for each hypothesis
    :return: A generator function that makes a prediction for a given example
    on all hypotheses, weighted by their respective values
    """
    def predict(example):
        return weighted((predictor(example) for predictor in predictors), weights)
    return predict

def weighted(values, weights):
    """
    Computes the maximum weight value
    :param values: Predicted label for example
    :param weights: Weight vector
    :return: Value with max weight (weighted plurality)
    """
    totals = defaultdict(int)
    for value, weight in zip(values, weights):
        totals[value] += weight
    return max(totals, key=totals.__getitem__)

def clip(x, lowest, highest):
    """
    Clips data so that no divide by zero error occurs
    :param x: input list
    :param lowest: smallest value
    :param highest: largest value
    :return: x - {lowest, highest}
    """
    return max(lowest, min(x, highest))

def normalize(dist):
    """
    Alpha normalization such that sum = 1.0
    :param dist: Given probability distribution
    :return: A normalized distribution
    """
    if isinstance(dist, dict):
        total = sum(dist.values())
        for key in dist:
            dist[key] = dist[key] / total
        return dist
    total = sum(dist)
    return [(n / total) for n in dist]