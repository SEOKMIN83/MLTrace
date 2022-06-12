# -*- coding: utf-8 -*-
"""
=====================
Classifier comparison
=====================

A comparison of a several classifiers in scikit-learn on synthetic datasets.
The point of this example is to illustrate the nature of decision boundaries
of different classifiers.
This should be taken with a grain of salt, as the intuition conveyed by
these examples does not necessarily carry over to real datasets.

Particularly in high-dimensional spaces, data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization than is achieved by other classifiers.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.

"""

# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay


## 10 classification algorithms
names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

# classifiers = [
#     KNeighborsClassifier(3),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     GaussianProcessClassifier(1.0 * RBF(1.0)),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1, max_iter=1000),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis(),
# ]

# X, y = make_classification(
#     n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
# )
# rng = np.random.RandomState(2)
# X += 2 * rng.uniform(size=X.shape)
# linearly_separable = (X, y)
#
# ## 3 datasets
# datasets = [
#     make_moons(noise=0.3, random_state=0),
#     make_circles(noise=0.2, factor=0.5, random_state=1),
#     linearly_separable,
# ]

# figure = plt.figure(figsize=(27, 9))
# i = 1
# iterate over datasets
# for ds_cnt, ds in enumerate(datasets):
#     # preprocess dataset, split into training and test part
#     X, y = ds
#     X = StandardScaler().fit_transform(X)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.4, random_state=42  ## train:60%, test:40%
#     )
#     # iterate over classifiers
#     for name, clf in zip(names, classifiers):  ## change multiple clasffication algorithms
#         # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
#         print(type(clf))
#         os.system('callgrind_control -i on')
#         clf.fit(X_train, y_train)  ## train section
#         os.system('callgrind_control -i off')
#         # score = clf.score(X_test, y_test)  ## test section
#         i += 1

if __name__ == '__main__':
    import os
    import sys

    '''
    sys.argv[1] - classifier : 1 - KNeighborsClassifier(3), ... 
    sys.argv[2] - data set : 1 - make_moons(noise=0.3, random_state=0), 2 - make_circles(noise=0.2, factor=0.5, random_state=1), 3 - linearly_separable 
    '''
    X, y = make_classification(
        n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    if int(sys.argv[2]) == 1:
        ds = make_moons(noise=0.3, random_state=0)
    elif int(sys.argv[2]) == 2:
        ds = make_circles(noise=0.2, factor=0.5, random_state=1)
    elif int(sys.argv[2]) == 3:
        ds = linearly_separable
    else:
        print("Data Set is Wrong.")
        exit()

    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42  ## train:60%, test:40%
    )

    classifiers_dict = {1: KNeighborsClassifier(3), 2: SVC(kernel="linear", C=0.025),
                        3: SVC(gamma=2, C=1), 4: GaussianProcessClassifier(1.0 * RBF(1.0)),
                        5: DecisionTreeClassifier(max_depth=5),
                        6: RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                        7: MLPClassifier(alpha=1, max_iter=1000),
                        8: AdaBoostClassifier(),
                        9:  GaussianNB(),
                        10: QuadraticDiscriminantAnalysis()}
    clf = classifiers_dict[int(sys.argv[1])]

    print('algorithm: %s, data set: %s, pid is %s' % (sys.argv[1], sys.argv[2], os.getpid()))
    # print('clf is %s' % clf)

    os.system('callgrind_control -i on')
    clf.fit(X_train, y_train)  ## train section
    os.system('callgrind_control -i off')
    # print("END")