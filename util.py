import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cross_validation import cross_val_predict
from sklearn.preprocessing import binarize

def benchmark(clfs, X_train, y_train, n_folds=6, show_report=False, threshold=0.5,
              metric = metrics.roc_auc_score, show_confusion=False):

    if type(clfs) is not list: clfs = [ clfs ]

    if show_confusion:
        print('=' * 80)

    scores = []

    for clf in clfs:

        if type(clf) == GridSearchCV:

            clf.fit(X_train, y_train)
            '''print("Best parameters set found on training set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on training set:")
            print()
            for params, mean_score, clf_scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, clf_scores.std() * 2, params))'''
            scores.append(clf.best_score_)
            pred_prob_1 = clf.predict_proba(X_train)[:, 1]
            predicted = binarize([pred_prob_1], threshold)[0]
            #predicted = clf.predict(X_train)

        else:
            predicted = cross_val_predict(clf, X_train, y_train,
                                                           cv=n_folds)
            clf_score = metric(y_train, predicted)
            #clf_score = cross_validation.cross_val_score(clf, X_train, y_train,
            #                                               cv=n_folds, scoring='f1')
            scores.append(np.mean(clf_score))

        if show_report:
            print("Classification report for classifier %s:\n%s"
              % (clf, metrics.classification_report(y_train, predicted)))

        if show_confusion:
            print("Confusion matrix:\n%s\n" % metrics.confusion_matrix(y_train, predicted))

    return scores

def visualize(X, y):

    X_tsne = TSNE(learning_rate=1000, random_state=17).fit_transform(X)
    X_pca = PCA(n_components=2).fit_transform(X)

    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.title("PCA")
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=50, cmap=plt.cm.Paired)
    plt.subplot(122)
    plt.title("t-SNE")
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, s=50, cmap=plt.cm.Paired)

def downsample(data_imbalanced, label, factor):
    train_class1 = train_imbalanced[data_imbalanced[label] == 1]
    train_class0 = train_imbalanced[train_imbalanced[label] == 0].sample(
        n=train_class1.shape[0] * factor, random_state=17, replace=True)
    return pd.concat([train_class0, train_class1]).sample(frac=1)

def get_temporal_folds(train):
    folds = [(train[(train.year == 2015) & (train.month <= 4)],
              train[(train.year == 2015) & (train.month == 5)]),
             (train[(train.year == 2015) & (train.month <= 6)],
              train[(train.year == 2015) & (train.month == 7)]),
             (train[(train.year == 2015) & (train.month <= 8)],
              train[(train.year == 2015) & (train.month == 9)]),
             (train[(train.year == 2015) & (train.month <= 10)],
              train[(train.year == 2015) & (train.month == 11)]),
             (train[(train.year == 2015) & (train.month <= 12)],
              train[(train.year == 2016) & (train.month == 1)]),
             (train[(train.year == 2015) | ((train.year == 2016) & (train.month <= 2))],
                  train[(train.year == 2016) & (train.month == 3)])]
    return folds

def isstring(s):
    return isinstance(s, str)

def fix_unicode(x):

    if not isstring(x):
        return unicode(x).encode('UTF-8')

    return x

def make_nonnegative(x):

    if x < 0:
        return 0
    else:
        return x
