# -*- coding: utf-8 -*-
from face_recognition import process_image as pi
from face_recognition import classifier as clf
import csv


def train(training_dir, fr=None, image_size=144, process_output=True,
          all_data=False, use_alignment=True, clf_output=True,
          language='chinese'):
    """
    Inputs:
      - train_dir: The directory of the training set.
      - fr: The object of the UI class.
      - image_size: The image size of the MTCNN network.
      - process_output: True or False, whether to supervise the
            training process.
      - all_data: True or False, whether to use all training data
            to train the classifier.
      - use_alignment: True or False, whether to use face alignment.
      - clf_output: True or False, whether to output the answer for
            each classifier.
      - language: 'chinese' or 'english'.

    Returns:
      - features: A numpy array with shape of (N, D) contains the
            features of all training images.
      - labels: A numpy array with shape of (N, ) contains the labels
            of all training images.
      - best_model_list: A list contains the best three model.
      - best_probability_model: The best softmax model.
    """

    features, labels = pi.process('training', training_dir,
                                  fr=fr,
                                  image_size=image_size,
                                  output=process_output,
                                  use_alignment=use_alignment,
                                  language=language)

    info_dict = {'param': 'value',
                 'training_dir': training_dir,
                 'image_size': image_size,
                 'use_alignment': use_alignment,
                 'all_data': all_data}
    if fr is None:
        write_list_or_dict_into_csv(info_dict, csv_path='models/info.csv')
    else:
        write_list_or_dict_into_csv(info_dict, csv_path='../models/info.csv')
    # return None, None, None, None
    # train best classification models.
    knn_model_list = clf.training_knn_classifier(features,
                                                 labels,
                                                 output=False)
    svm_model_list = clf.training_svm_classifier(features,
                                                 labels,
                                                 output=False)
    softmax_model_list = clf.training_softmax_classifier(features,
                                                         labels,
                                                         output=False)

    best_classifier_model_list = clf.get_best_model(
        knn_model_list, svm_model_list, softmax_model_list)
    clf.save_best_classifier_model(best_classifier_model_list,
                                   all_data=False)

    # train best probability model.
    best_probability_model = clf.get_best_probability_model(softmax_model_list)
    clf.save_best_probability_model(best_probability_model,
                                    all_data=False)

    if all_data:
        best_classifier_model_list = train_model_by_all_data(features,
                                                             labels,
                                                             best_classifier_model_list)
        best_probability_model = train_model_by_all_data(features,
                                                         labels,
                                                         best_probability_model,
                                                         probability=True)

    if clf_output:
        model_info = []
        model_info = check_model(fr,
                                 model_info=model_info,
                                 probability=False,
                                 all_data=all_data,
                                 language=language)
        model_info = check_model(fr,
                                 model_info=model_info,
                                 probability=True,
                                 all_data=all_data,
                                 language=language)
    return features, labels, best_classifier_model_list, best_probability_model


def write_list_or_dict_into_csv(data, have_chinese=False, csv_path=None):
    """Write the answer into a csv file

    Inputs:
      - data: A list or dict.
      - have_chinese: True or False, whether the data contains chinese.
      - csv_path: A string which contains the path to the csv file.
    """
    if type(data) not in {list, dict}:
        raise ValueError('The type of the input data must '
                         'be \'dict\' or \'list\'')

    if csv_path is None:
        raise ValueError('csv_path should not be \'None\'')
    else:
        if have_chinese:
            encoding = 'utf-8-sig'
        else:
            encoding = 'utf-8'
        with open(csv_path, 'w', newline='', encoding=encoding) as f:
            w = csv.writer(f)
            if type(data) is dict:
                for key, val in data.items():
                    w.writerow([key, val])
            else:
                w.writerows(data)


def train_model_by_all_data(features, labels,
                            best_model_trained_by_partial_data,
                            probability=False):
    """ Uses all training data to fit model.
    Inputs:
      - features: A numpy array with shape of (N, D) contains the
            features of all training images.
      - labels: A numpy array with shape of (N, ) contains the
            labels of all training images.
      - best_model_list_trained_by_partial_data: A list contains
            the best three models which trained by partial data
            of the training set.

    Return:
      - best_model_list_trained_by_all_data: A list contains the
            best three models which trained by all training data.
    """
    if probability:
        best_probability_model_trained_by_all_data = clf.fit_model_by_all_data(features,
                                                                               labels,
                                                                               best_model_trained_by_partial_data,
                                                                               probability=True)
        clf.save_best_probability_model(best_probability_model_trained_by_all_data,
                                        all_data=True)
        return best_probability_model_trained_by_all_data

    else:
        best_classifier_model_list_trained_by_all_data = clf.fit_model_by_all_data(features,
                                                                                   labels,
                                                                                   best_model_trained_by_partial_data)
        clf.save_best_classifier_model(best_classifier_model_list_trained_by_all_data,
                                       all_data=True)
        return best_classifier_model_list_trained_by_all_data


def check_model(fr, model_info=None, probability=False, all_data=False,
                language='chinese'):
    """Checks best models' accuracy or parameters.

    Inputs:
      - fr: The object of the UI class.
      - model_info: A string which contains the information of the model.
      - probability: True or False, whether to use softmax probability model.
      - all_data: True or False, whether the model is trained with all data.
      - language: 'chinese' or 'english'.

    Return:
      - model_info: A string which contains the information of the model.
    """
    if probability:
        if all_data is False:
            if language == 'chinese':
                info = '通过训练集部分数据得到的概率模型：'
            else:
                info = 'Probability model trained by partial data of the training set:'
            print(info)
            model_info.append(info)
            if fr is not None:
                fr.show_information(model_info, clf_output=True)
            best_probability_model = clf.load_best_probability_model(all_data=all_data)
            
            if language == 'chinese':
                info = '模型类别：{}'.format(best_probability_model[4])
            else:
                info = 'Type of model: {}'.format(best_probability_model[4])
            print(info)
            model_info.append(info)

            if language == 'chinese':
                info = '训练集准确度：{:.2%}，验证集大小为：{}, 验证集准确度：{:.2%}'.format(
                        best_probability_model[1], best_probability_model[3], best_probability_model[2])
            else:
                info = 'Accuracy on the training set: {:.2%}, size of the validation set: {}, ' \
                           'accuracy on the validation set: {:.2%}.'.format(best_probability_model[1],
                                                                            best_probability_model[3],
                                                                            best_probability_model[2])
            print(info)
            model_info.append(info)
            if fr is not None:
                fr.show_information(model_info, clf_output=True)

            # print(best_probability_model)
            model_info.append('')
            if fr is not None:
                fr.show_information(model_info, clf_output=True)

        else:
            if language == 'chinese':
                info = '通过训练集全部数据得到的概率模型：'
            else:
                info = 'Probability model trained by all data of the training set:'
            print(info)
            model_info.append(info)
            if fr is not None:
                fr.show_information(model_info, clf_output=True)

            best_probability_model = clf.load_best_probability_model(all_data=all_data)
            if language == 'chinese':
                info = '模型类别：{}'.format(best_probability_model[4])
            else:
                info = 'Type of model: {}'.format(best_probability_model[4])
            print(info)
            model_info.append(info)
            if language == 'chinese':
                info = '训练集准确度：{:.2%}，验证集大小为：{}，' \
                       '验证集准确度：{:.2%}'.format(best_probability_model[1],
                                              best_probability_model[3],
                                              best_probability_model[2])
            else:
                info = 'Accuracy on the training set: {:.2%}, size of the validation set: {},' \
                       'accuracy on the validation set: {:.2%}.'.format(best_probability_model[1],
                                                                        best_probability_model[3],
                                                                        best_probability_model[2])
            print(info)
            model_info.append(info)
            if fr is not None:
                fr.show_information(model_info, clf_output=True)
            # print(best_probability_model)
            # print(model_info)
            model_info.append('')
            if fr is not None:
                fr.show_information(model_info, clf_output=True)

    else:
        if all_data is False:
            if language == 'chinese':
                info = '通过训练集部分数据得到的识别模型：'
            else:
                info = 'Recognition model trained by partial data of the training set:'
            print(info)
            model_info.append(info)
            if fr is not None:
                fr.show_information(model_info, clf_output=True)

            best_classifier_model_list = clf.load_best_classifier_model(all_data=all_data)

            for index, model in enumerate(best_classifier_model_list):
                if language == 'chinese':
                    info = '模型类别：{}'.format(model[4])
                else:
                    info = 'Type of model: {}'.format(model[4])
                print(info)
                model_info.append(info)
                if language == 'chinese':
                    info = '训练集准确度：{:.2%}，验证集大小为：{}，' \
                           '验证集准确度：{:.2%}'.format(model[1], model[3], model[2])
                else:
                    info = 'Accuracy on the training set: {:.2%}, size of the validation set: {}, ' \
                           'accuracy on the validation set: {:.2%}'.format(model[1], model[3], model[2])
                print(info)
                model_info.append(info)
                if fr is not None:
                    fr.show_information(model_info, clf_output=True)
                # print(model)
                # print(model_info)
            model_info.append('')
            if fr is not None:
                fr.show_information(model_info, clf_output=True)

        else:
            if language == 'chinese':
                info = '通过训练集全部数据得到的识别模型：'
            else:
                info = 'Recognition model trained by all data of the training set:'
            print(info)
            model_info.append(info)
            if fr is not None:
                fr.show_information(model_info, clf_output=True)

            best_classifier_model_list = clf.load_best_classifier_model(all_data=all_data)
            for index, model in enumerate(best_classifier_model_list):
                # print(model)
                if language == 'chinese':
                    info = '模型类别：{}'.format(model[4])
                else:
                    info = 'Type of model: {}'.format(model[4])
                print(info)
                model_info.append(info)
                if language == 'chinese':
                    info = '训练集准确度：{:.2%}，验证集大小为：{}，' \
                           '验证集准确度：{:.2%}'.format(model[1], model[3], model[2])
                else:
                    info = 'Accuracy on the training set: {:.2%}, size of the validation set: {}, ' \
                           'accuracy on the validation set: {:.2%}'.format(model[1], model[3], model[2])
                print(info)
                model_info.append(info)
                if fr is not None:
                    fr.show_information(model_info, clf_output=True)
                # print(model)
            model_info.append('')
            if fr is not None:
                fr.show_information(model_info, clf_output=True)

    return model_info
