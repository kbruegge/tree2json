from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def evaluate_tree(clf, sample):
    '''
    Return the predictions from a decissiontree.
    '''
    sample = np.ravel(sample)
    threshold = clf.tree_.threshold
    attribute = clf.tree_.feature
    left = clf.tree_.children_left
    right = clf.tree_.children_right
    dist = clf.tree_.value

    index = 0
    while attribute[index] >= 0:
        split_attribute = attribute[index]
        value = sample[split_attribute]
        node_threshold = threshold[index]
        left_node = left[index]
        right_node = right[index]

        value = sample[split_attribute]

        if value < node_threshold:
            index = left_node
        else:
            index = right_node

    return dist[index]


def main():
    '''
    Build simple model to test whether the evaluation algorithm is correct
    by comparing output from evaluate_tree to output from  clf.predict_proba
    '''
    X, y = load_iris(return_X_y=True)
    clf = DecisionTreeClassifier(random_state=1234)
    clf.fit(X, y)

    for i in range(150):
        sample = X[i].reshape(1, -1)
        p = clf.predict_proba(sample)
        p_m = evaluate_tree(clf, sample)
        assert p.argmax() == p_m.argmax(), 'prediction from eval method should be equal to the sklearn one'


if __name__ == '__main__':
    main()
