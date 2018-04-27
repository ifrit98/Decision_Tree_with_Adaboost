from math import log2
from ada import adaboost
import random
import pickle

class DecisionTreeNode:

    def __init__(self, feature, default_child=None, children=None):
        self.feature = feature
        self.default_child = default_child
        self.children = children or {}

    def __call__(self, example):
        # Recursive classifier
        feature_val = example[self.feature]
        if feature_val in self.children:
            return self.children[feature_val](example)
        else:
            return self.default_child(example)

    def add_subtree(self, value, subtree):
        self.children[value] = subtree

class DecisionTreeLeaf:

    def __init__(self, label):
        self.label = label

    def __call__(self, example):
        return self.label

def decision_tree(data, weights=None):

    label, values, weight = data.label, data.values, data.weight

    if not weights: # Create flat weight distribution for first iteration (1-level stump)
        weights = [1/len(data.examples) for ex in data.examples]

    for i in range(len(data.examples)): # Insert weights at end of example (pre-allocated)
        data.examples[i].insert(-1,weights[i])

    def train_tree(examples, features, parent_ex=(), depth=1):
        if len(examples) == 0 or depth == 0: # If we exceeded max_depth or there are no more examples to classify
            return weighted_majority(parent_ex)
        elif all_same_class(examples): # If all the examples are the same language
            return DecisionTreeLeaf(examples[0][label])
        elif len(features) < 1: # If we ran out of features to use
            return weighted_majority(examples)
        else:
            FEATURE = choose_feature(features, examples)
            tree = DecisionTreeNode(FEATURE, weighted_majority(examples))
            for (v_k, ex) in split(FEATURE, examples):
                subtree = train_tree(ex, remove_all(FEATURE, features), examples, depth-1)
                tree.add_subtree(v_k, subtree)

        return tree

    def weighted_majority(examples):
        nl = sum([ex[weight] for ex in examples if ex[label] == 'nl'])
        en = sum([ex[weight] for ex in examples if ex[label] == 'en'])
        wt = [(nl,'nl'),(en,'en')]
        m =  max(wt, key=lambda x: x[0])

        return DecisionTreeLeaf(m[1])

    def choose_feature(features, examples):
        return max_tie(features, key=lambda f: information_gain(f, examples))

    def information_gain(feature, examples):
        if not isinstance(examples[0][feature],bool): # Avoid weights and labels being used as features
            return 0
        groupT = [ex for ex in examples if ex[feature] == True]
        groupF = [ex for ex in examples if ex[feature] == False]
        p1 = sum([ex[weight] for ex in groupT if ex[label] == 'nl'])
        p2 = sum([ex[weight] for ex in groupF if ex[label] == 'nl'])
        n1 = sum([ex[weight] for ex in groupT if ex[label] == 'en'])
        n2 = sum([ex[weight] for ex in groupF if ex[label] == 'en'])

        # Avoid divide by zero error
        if p1+n1 == 0 or p2+n2 == 0:
            return 1.0
        P = p1+p2
        N = p1+p2+n1+n2

        def B(q):
            if q == 1.0 or q == 0:
                return 0
            return -(q*log2(q) + (1-q)*log2(1-q))

        remainder = ((p1+n1) / N * B(p1/(p1+n1))) + ((p2+n2) / N * B(p2/(p2+n2)))
        return B(P/N) - remainder

    def all_same_class(exs):
        language = exs[0][label]
        return all(x[label] == language for x in exs)

    def split(feature, examples):
        return [(value, [example for example in examples if example[feature] == value])
                for value in values[feature]]

    return train_tree(data.examples, data.inputs, depth=3)


def max_tie(seq, key=lambda x: x):
    return max(shuffle(seq), key=key)

def shuffle(iterable):
    items = list(iterable)
    random.shuffle(items)
    return items

def remove_all(item, seq):
    if isinstance(seq, str):
        return seq.replace(item, '')
    else:
        return [x for x in seq if x != item]


def train(data, hypOut='hypothesis_out.txt', learning_type='dt'):
    """
    Selector algorithm that calls the proper training model and
    pickles (serializes) the model that is returned.
    :param data: Input dataset to be trained on
    :param hypOut: Output filename for pickled model
    :param learning_type: Adaboost or Decision Tree
    :return: None
    """
    if learning_type == 'ada':
        K = 10  # Number of stumps to create (adaboost iterations)
        ada_train = adaboost(decision_tree, K)
        boosted_tree = ada_train(data)
        # Pickles boosted_tree
        hypothesis_file = open(hypOut, 'wb')
        pickle.dump(boosted_tree, hypothesis_file)
        hypothesis_file.close()

    else:
        tree = decision_tree(data)
        # Pickles tree
        hypothesis_file = open(hypOut, 'wb')
        pickle.dump(tree, hypothesis_file)
        hypothesis_file.close()
