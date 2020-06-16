from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt


def selector(n, features, labels, method):
    chi2_selector = SelectKBest(method, k=n)
    return chi2_selector.fit_transform(features, labels)


def examine_features(nfeatures, labels, select_method, features, model, folds=5):
    cv_scores_list = []
    for n in nfeatures:
        cv_scores = cross_val_score(model, selector(n, features, labels, select_method), labels, cv=folds,
                                    scoring='accuracy')
        cv_scores_list.append(cv_scores.max())
    cv_scores = np.array(cv_scores_list)
    return cv_scores


def examine_tree_depth(tree_depths, labels, feature_selector, folds=5):
    cv_scores_list = []
    for depth in tree_depths:
        tree_model = DecisionTreeClassifier(max_depth=depth)
        cv_scores = cross_val_score(tree_model, feature_selector, labels, cv=folds, scoring='accuracy')
        cv_scores_list.append(cv_scores.max())
    cv_scores_arr = np.array(cv_scores_list)
    return cv_scores_arr


def examine_tree_pruning(prunings, labels, feature_selector):
    cv_scores_list = []
    for prune in prunings:
        tree_model = DecisionTreeClassifier(ccp_alpha=prune)
        cv_scores = cross_val_score(tree_model, feature_selector, labels, cv=5, scoring='accuracy')
        cv_scores_list.append(cv_scores.max())
    cv_scores_arr = np.array(cv_scores_list)
    return cv_scores_arr


def plot_accuracy(cv_scores, title, x_label, x_values):
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(x_values, cv_scores, '-o', label='accuracy', alpha=0.9)
    ylim = plt.ylim()
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(x_values)
    ax.legend()
