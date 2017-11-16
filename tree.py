from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def evaluate_tree(clf, sample):
    sample = np.ravel(sample)
    # print(sample)

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

        # print('index {}, Split Attribute {}, node_threshold {}, value {}, left_node {}, right_node{}'
        #         .format(index, split_attribute, node_threshold, value, left_node, right_node))
        value = sample[split_attribute]

        if value < node_threshold:
            index = left_node
        else:
            index = right_node

    return dist[index]


X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(random_state=1234)

clf.fit(X, y)

for i in range(150):
    sample = X[i].reshape(1, -1)
    print(sample)
    p = clf.predict_proba(sample)
    p_m = evaluate_tree(clf, sample)
    print(p)

    assert p.argmax() == p_m.argmax()
