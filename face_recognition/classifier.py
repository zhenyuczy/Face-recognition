# -*- coding: utf-8 -*-

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from operator import itemgetter
from sklearn.linear_model import LogisticRegression
import os


def training_knn_classifier(features, labels, flag=None, output=False):
    """
    Inputs:
      - features: A numpy array with shape of (N, D) contains the features of all
              training images.
      - labels: A numpy array with shape of (N, ) contains the labels of all training
              images.
      - flag: The neighbors of the knn-classifier. If flag is None, we need
              to split the data to get the best classifier.
      - output: Print the classifiers' accuracy.
    Returns:
      - return a list contains (model, train_accuracy, test_accuracy, clf_name, parameter)

    """
    if flag is None:
        skf = StratifiedKFold(n_splits=3)
        split_index = 0
        knn_model_list = []
        for train, test in skf.split(features, labels):
            X_train = features[train]
            y_train = labels[train]
            X_test = features[test]
            y_test = labels[test]
            neighbors = [1, 2, 3, 4, 5, 6, 7]
            split_model_list = []
            for nb in neighbors:
                if nb > X_train.shape[0]:
                    continue
                if nb > X_test.shape[0]:
                    continue
                knn_clf = KNeighborsClassifier(n_neighbors=nb)
                model = knn_clf.fit(X_train, y_train)
                if output is True:
                    print('split: {}, train size: {}, test_size: {}, neighbor: {}'.format(
                        split_index, X_train.shape[0], X_test.shape[0], nb))

                train_predict = model.predict(X_train)
                train_accuracy = metrics.accuracy_score(y_train, train_predict)
                if output is True:
                    print('train accuracy: %.2f%%' % (100 * train_accuracy))

                test_predict = model.predict(X_test)
                test_accuracy = metrics.accuracy_score(y_test, test_predict)

                if output is True:
                    print('test accuracy: %.2f%%' % (100 * test_accuracy))
                    print()

                split_model_list.append((model, train_accuracy, test_accuracy, X_test.shape[0], 'knn', nb))

            knn_model_list.append(sorted(split_model_list, key=itemgetter(2, 1, 3), reverse=True)[0])
            split_index += 1

        if split_index > 3:
            knn_model_list = sorted(knn_model_list, key=itemgetter(2, 1, 3), reverse=True)[0:3]

        return knn_model_list

    else:
        knn_model = KNeighborsClassifier(n_neighbors=flag).fit(features, labels)
        return (knn_model)


def training_svm_classifier(features, labels, flag=None, output=False):
    if flag is None:
        skf = StratifiedKFold(n_splits=3)
        split_index = 0
        svm_model_list = []

        for train, test in skf.split(features, labels):

            X_train = features[train]
            y_train = labels[train]
            X_test = features[test]
            y_test = labels[test]

            regularization_length = [1, 2, 3, 4, 5, 6, 7]
            split_model_list = []
            for rl in regularization_length:
                model = svm.LinearSVC(C=rl).fit(X_train, y_train)

                if output is True:
                    print('split: {}, train size: {}, test_size: {}, rl: {}'.format(
                        split_index, X_train.shape[0], X_test.shape[0], rl))

                train_predict = model.predict(X_train)
                train_accuracy = metrics.accuracy_score(y_train, train_predict)

                if output is True:
                    print('train accuracy: %.2f%%' % (100 * train_accuracy))

                test_predict = model.predict(X_test)
                test_accuracy = metrics.accuracy_score(y_test, test_predict)

                if output is True:
                    print('test accuracy: %.2f%%' % (100 * test_accuracy))
                    print()

                split_model_list.append((model, train_accuracy, test_accuracy, X_test.shape[0], 'svm', rl))

            svm_model_list.append(sorted(split_model_list, key=itemgetter(2, 1, 3), reverse=True)[0])
            split_index += 1

        if split_index > 3:
            svm_model_list = sorted(split_model_list, key=itemgetter(2, 1, 3), reverse=True)[0:3]

        return svm_model_list

    else:
        svm_model = svm.LinearSVC(C=flag).fit(features, labels)
        return (svm_model)


def training_softmax_classifier(features, labels, flag=None, output=False):
    if flag is None:
        skf = StratifiedKFold(n_splits=3)
        split_index = 0
        softmax_model_list = []

        for train, test in skf.split(features, labels):

            X_train = features[train]
            y_train = labels[train]
            X_test = features[test]
            y_test = labels[test]

            regularization_length = [1, 2, 3, 4, 5, 6, 7]
            split_model_list = []
            for rl in regularization_length:
                model = LogisticRegression(
                    C=rl, multi_class='multinomial', solver='sag').fit(X_train, y_train)

                if output is True:
                    print('split: {}, train size: {}, test_size: {}, rl: {}'.format(
                        split_index, X_train.shape[0], X_test.shape[0], rl))

                train_predict = model.predict(X_train)
                train_accuracy = metrics.accuracy_score(y_train, train_predict)

                if output is True:
                    print('train accuracy: %.2f%%' % (100 * train_accuracy))

                test_predict = model.predict(X_test)
                test_accuracy = metrics.accuracy_score(y_test, test_predict)

                if output is True:
                    print('test accuracy: %.2f%%' % (100 * test_accuracy))
                    print()

                split_model_list.append((model, train_accuracy, test_accuracy, X_test.shape[0], 'softmax', rl))

            split_index += 1
            softmax_model_list.append(sorted(split_model_list, key=itemgetter(2, 1, 3), reverse=True)[0])

        if split_index > 3:
            softmax_model_list = sorted(softmax_model_list, key=itemgetter(2, 1, 3), reverse=True)[0:3]

        return softmax_model_list
    else:
        softmax_model = LogisticRegression(
            C=flag, multi_class='multinomial', solver='sag').fit(features, labels)
        return (softmax_model)


def get_best_model(knn_model_list, svm_model_list, softmax_model_list):
    """ Gets the best models from knn_models, svm_models and softmax_models.

    Inputs:
      - knn_model_list: Top-3 knn classifiers.
      - svm_model_list: Top-3 svm classifiers.
      - softmax_model_list: Top-3 softmax classifiers.

    Return:
      - best_model_list: Top-3 classifiers.
    """
    best_model_list = []
    for i in range(len(knn_model_list)):
        best_model_list.append(knn_model_list[i])
        best_model_list.append(svm_model_list[i])
        best_model_list.append(softmax_model_list[i])
    if len(best_model_list) > 3:
        best_model_list = sorted(best_model_list, key=itemgetter(2, 1, 3), reverse=True)[0:3]

    return best_model_list


def get_best_probability_model(softmax_model_list):
    best_probability_model = sorted(
        softmax_model_list, key=itemgetter(2, 1, 3), reverse=True)[0]
    return best_probability_model


def fit_model_by_all_data(features, labels, model_list, probability=False):
    """ Uses all training data to fit models.

    Inputs:
      - features: A numpy array with shape of (N, D) contains the features of all
              training images.
      - labels: A numpy array with shape of (N, ) contains the labels of all training
              images.
      - model_list: A list contains the models which trained by partial training data.

    Return:
      - new_model_list: A list contains the models which trained by all training data.
    """
    if probability:
        model_list = (model_list[0].fit(features, labels),
                      model_list[1], model_list[2], model_list[3],
                      model_list[4], model_list[5])
        return model_list

    new_model_list = []
    for index, model in enumerate(model_list):
        if model[4] == 'knn':
            new_model_list.append((training_knn_classifier(
                features, labels, flag=model[5]), model[1],
                model[2], model[3], model[4], model[5]))

        elif model[4] == 'svm':
            new_model_list.append((training_svm_classifier(
                features, labels, flag=model[5]), model[1],
                model[2], model[3], model[4], model[5]))

        else:
            new_model_list.append((training_softmax_classifier(
                features, labels, flag=model[5]), model[1],
                model[2], model[3], model[4], model[5]))

    return new_model_list


def save_best_probability_model(best_probability_model, all_data=False):
    # save model
    if all_data:
        if os.path.exists('models'):
            joblib.dump(best_probability_model,
                        'models/my_models/best_probability_model_trained_by_all_data.pkl')
        else:
            joblib.dump(best_probability_model,
                        '../models/my_models/best_probability_model_trained_by_all_data.pkl')
    else:
        if os.path.exists('models'):
            joblib.dump(best_probability_model,
                        'models/my_models/best_probability_model_trained_by_partial_data.pkl')
        else:
            joblib.dump(best_probability_model,
                        '../models/my_models/best_probability_model_trained_by_partial_data.pkl')


def save_best_classifier_model(best_classifier_model_list, all_data=False):
    # save model
    if all_data:
        if os.path.exists('models'):
            joblib.dump(best_classifier_model_list,
                        'models/my_models/best_classifier_model_list_trained_by_all_data.pkl')
        else:
            joblib.dump(best_classifier_model_list,
                        '../models/my_models/best_classifier_model_list_trained_by_all_data.pkl')
    else:
        if os.path.exists('models'):
            joblib.dump(best_classifier_model_list,
                        'models/my_models/best_classifier_model_list_trained_by_partial_data.pkl')
        else:
            joblib.dump(best_classifier_model_list,
                        '../models/my_models/best_classifier_model_list_trained_by_partial_data.pkl')


def load_best_probability_model(all_data=False):
    if all_data:
        if os.path.exists('models'):
            best_probability_model = joblib.load(
                'models/my_models/best_probability_model_trained_by_all_data.pkl')
        else:
            best_probability_model = joblib.load(
                '../models/my_models/best_probability_model_trained_by_all_data.pkl')
    else:
        if os.path.exists('models'):
            best_probability_model = joblib.load(
                'models/my_models/best_probability_model_trained_by_partial_data.pkl')
        else:
            best_probability_model = joblib.load(
                '../models/my_models/best_probability_model_trained_by_partial_data.pkl')
    return best_probability_model


def load_best_classifier_model(all_data=False):
    if all_data:
        if os.path.exists('models'):
            best_classifier_model_list = joblib.load(
                'models/my_models/best_classifier_model_list_trained_by_all_data.pkl')
        else:
            best_classifier_model_list = joblib.load(
                '../models/my_models/best_classifier_model_list_trained_by_all_data.pkl')
    else:
        if os.path.exists('models'):
            best_classifier_model_list = joblib.load(
                'models/my_models/best_classifier_model_list_trained_by_partial_data.pkl')
        else:
            best_classifier_model_list = joblib.load(
                '../models/my_models/best_classifier_model_list_trained_by_partial_data.pkl')
    return best_classifier_model_list
