# -*- coding: utf-8 -*-

from face_recognition import process_image as pi
from face_recognition import classifier as clf
import numpy as np
import cv2
from skimage import io
from PIL import Image, ImageDraw, ImageFont
import os


def predict_labels(best_model_list,
                   test_features,
                   face_number,
                   all_data=False):
    """Predict with the best three models.
    Inputs:
      - best_model_list: A list contains the best three models.
      - test_features: A numpy array contains the features of the test images.
      - face_number: An integer, the number of faces that we found in the test image.
      - all_data: True or False, whether the model is trained with all data.

    Return:
      - labels_array: A numpy array contains the labels which predicted by
            the best three models .
    """
    labels_array = []
    for index, model in enumerate(best_model_list):
        if all_data:
            predict = model[0].predict(test_features)
        else:
            predict = model[0].predict(test_features)
        labels_array.append(predict)

    labels_array = np.array(labels_array)
    return labels_array


def predict_probability(best_probability_model,
                        test_features,
                        face_number,
                        all_data=False):
    """Predict with softmax probability model.
    Inputs:
      - best_probability_model: The best pre_trained softmax model.
      - test_features: A numpy array contains the features of the test images.
      - face_number: An integer, the number of faces that we found in the test image.
      - all_data: True or False, whether the model is trained with all data.

    Returns:
      - labels: A numpy array contains the predicted labels.
      - probability: A numpy array contains the probability of the label.
    """
    labels, probability = None, None
    if face_number is not 0:
        if all_data:
            labels = best_probability_model[0].predict(test_features)
            all_probability = best_probability_model[0].predict_proba(test_features)
        else:
            labels = best_probability_model[0].predict(test_features)
            all_probability = best_probability_model[0].predict_proba(test_features)

        # print(probability.shape) -> (face_number, class_number)
        # print(all_probability)
        probability = all_probability[np.arange(face_number),
                                      np.argmax(all_probability, axis=1)]

    return labels, probability


def find_people_from_image(best_probability_model, test_features, face_number, all_data=False):
    """
    Inputs:
      - best_probability_model: The best pre_trained softmax model.
      - test_features: A numpy array contains the features of the test images.
      - face_number: An integer, the number of faces that we found in the test image.
      - all_data: If all_data is True, using the models which trained by all
          training data. Otherwise using the models which trained by partial data.

    e.g.
    test_features: [feature_1, feature_2, feature_3, feature_4, feature_5]

    first we get all predictive labels and their corresponding probability
    [(A, 0.3), (A, 0.2), (B, 0.1), (B, 0.3), (C, 0.1)]
    second we choose the maximum probability for each unique label
    [(A, 0.3), (B, 0.3), (C, 0.1)]
    finally, get the indices for each unique label.
    [0, 3, 4]

    then return
    label:
    probability:
    unique_labels: [A, B, C]
    unique_probability: [0.3, 0.3, 0.1]
    unique_indices: [0, 3, 4]
    """
    labels, probability = predict_probability(
        best_probability_model, test_features, face_number, all_data=all_data)

    unique_labels = np.unique(labels)
    unique_probability, unique_indices = [], []

    for label in unique_labels:
        indices = np.argwhere(labels == label)[:, 0]
        unique_probability.append(np.max(probability[indices]))
        unique_indices.append(indices[np.argmax(probability[indices])])

    unique_probability = np.array(unique_probability)
    unique_indices = np.array(unique_indices)

    return labels, probability, unique_labels, unique_probability, unique_indices


def check_sf_features(feature, label):
    """Verification.
    Inputs:
       - feature: The feature to be verified.
       - label: The label used for verification.

    Returns:
       - True or False, verification result.
       - sum_dis, the distance loss.
    """
    if os.path.exists('features'):
        sf_features = np.load(open('features/{}.npy'.format(label), 'rb'))
    else:
        sf_features = np.load(open('../features/{}.npy'.format(label), 'rb'))
    sf_dis = np.sqrt(np.sum((sf_features-feature)**2, axis=1))
    # print(sf_dis)
    sum_dis = np.sum(sf_dis)
    # print(sum_dis)
    valid_num1 = np.sum(sf_dis > 1.0)
    valid_num2 = np.sum(sf_dis > 1.1)
    if valid_num1 >= 4:
        return False, sum_dis

    if valid_num2 >= 3:
        return False, sum_dis

    return True, sum_dis


def recognition(image_path, fr=None, state='recognition',
                used_labels=None, image_size=144,
                all_data=False, output=False, language='chinese'):
    """Implement face verification, face recognition and face search functions.
    Inputs:
      - image_path: A string contains the path to the image.
      - fr: The object of the UI class.
      - state: 'verification', face verification
               'recognition', face recognition
               'search', face search.
      - used_labels: The labels used to face verification and face search,
            which does not used in face recognition
      - image_size: The input size of the MTCNN.
      - all_data: If all_data is True, using the models which trained by all
            training data. Otherwise using the models which trained by partial data.
      - output: True or False, output the process information.

    Returns:
      - answer: The answer predicted by the model.
      - image_data: The image data after prediction.
    """
    if state not in {'verification', 'recognition', 'search'}:
        raise ValueError('{} is not a valid argument!'.format(state))

    test_features, face_number, face_boxes = pi.process(
        'test', image_path, fr=fr, image_size=image_size, output=output)

    answer = ''
    image_data = io.imread(image_path)

    predict_info = []

    '''    
    if state == 'recognition':
        answer = answer + '从这个图像中检测出{}张人脸.\n'.format(face_number)
        info = '从这个图像中检测出{}张人脸.'.format(face_number)
        if fr is not None:
            predict_info.append(info)
            fr.show_information(predict_info, predict=True)
        if output:
            print(info)
    '''
    if state == 'verification':
        best_classifier_model_list = clf.load_best_classifier_model(all_data=all_data)
        labels_array = predict_labels(best_classifier_model_list,
                                      test_features,
                                      face_number,
                                      all_data=all_data)
        labels = []
        for line in labels_array.T:
            unique, counts = np.unique(line, return_counts=True)
            temp_label = unique[np.argmax(counts)]
            labels.append(temp_label)

        if used_labels[0] in labels:
            if language == 'chinese':
                answer = answer + '验证成功！这张图像被认定为{}！'.format(used_labels[0])
                info = '验证成功！这张图像被认定为{}！'.format(used_labels[0])
            else:
                answer = answer + 'Successful Verification! This image was ' \
                                  'identified as {}!'.format(used_labels[0])
                info = 'Successful Verification! This image ' \
                       'was identified as {}!'.format(used_labels[0])
        else:
            if language == 'chinese':
                answer = answer + '验证失败！这张图像不被认定为{}！' \
                                  ''.format(used_labels[0])
                info = '验证失败！这张图像不被认定为{}！'.format(used_labels[0])
            else:
                answer = answer + 'Verification failed! This image is not ' \
                                  'recognized as {}!'.format(used_labels[0])
                info = 'Verification failed! This image is not ' \
                       'recognized as {}!'.format(used_labels[0])

        for index, box in enumerate(face_boxes):

            face_position = box.astype(int)
            cv2.rectangle(image_data, (face_position[0], face_position[1]), (
                face_position[2], face_position[3]), (0, 255, 0), 8)

        if fr is not None:
            predict_info.append(info)
            fr.show_information(predict_info, predict=True)

    elif state == 'recognition':
        '''
        best_classifier_model_list = clf.load_best_classifier_model(all_data=all_data)
        labels_array = predict_labels(best_classifier_model_list, test_features,
                                      face_number, all_data=all_data)
        info = ''
        if language == 'chinese':
            info = info + '从图像中检测到'
        else:
            info = info + 'Detected '
        answer = answer + info
        labels = []
        for line, feature in zip(labels_array.T, test_features):
            unique, counts = np.unique(line, return_counts=True)
            temp_label = unique[np.argmax(counts)]

            if check_sf_features(feature, temp_label)[0] is False:
                if language == 'chinese':
                    temp_label = '未知'
                else:
                    temp_label = 'Unknown'
            labels.append(temp_label)

        labels = np.array(labels)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if language == 'chinese':
                if label == '未知':
                    continue
            else:
                if label == 'Unknown':
                    continue
            indices = np.argwhere(labels == label)[:, 0]
            all_sum_dis = []
            for index in indices:
                judge, sum_dis = check_sf_features(test_features[index], label)
                if judge is True:
                    all_sum_dis.append(sum_dis)
            if len(all_sum_dis) == 0:
                min_index = None
            else:
                min_index = indices[int(np.argmin(all_sum_dis))]
            if language == 'chinese':
                labels[indices] = '未知'
            else:
                labels[indices] = 'Unknown'
            if min_index is not None:
                labels[min_index] = label
                if language == 'chinese':
                    answer += label + '，'
                    info += label + '，'
                else:
                    answer += label + ','
                    info += label + ','

        answer = answer[:-1]
        info = info[:-1]
        if language == 'english':
            answer += ' in this image.'
            info += ' in this image.'
        else:
            answer += '。'
            info += '。'

        if fr is not None:
            predict_info.append(info)
            fr.show_information(predict_info, predict=True)

        if output:
            print(labels)

        for i in range(face_number):
            face_position = face_boxes[i].astype(int)
            label_pixels, font_size, line_size, rect_size, up_offset = \
                None, None, None, None, None
            if face_number == 1:
                rect_size = 8
                if language == 'chinese':
                    label_pixels = 160 * len(labels[i])
                    font_size = 160
                    up_offset = 160
                else:
                    label_pixels = 90 * len(labels[i])
                    up_offset = 20
                    font_size = 5
                line_size = 9
            elif face_number < 4:
                rect_size = 7
                if language == 'chinese':
                    label_pixels = 140 * len(labels[i])
                    font_size = 140
                    up_offset = 140
                else:
                    label_pixels = 70 * len(labels[i])
                    up_offset = 20
                    font_size = 4
                line_size = 7
            elif face_number >= 4:
                rect_size = 6
                if language == 'chinese':
                    label_pixels = 100 * len(labels[i])
                    font_size = 100
                    up_offset = 100
                else:
                    label_pixels = 50 * len(labels[i])
                    up_offset = 20
                    font_size = 3
                line_size = 5

        '''
        best_probability_model = clf.load_best_probability_model(all_data=all_data)
        labels, _, unique_labels, unique_probability, unique_indices = \
            find_people_from_image(best_probability_model, test_features, face_number, all_data=all_data)
        info = ''
        if language == 'chinese':
            info = info + '从图像中检测到'
        else:
            info = info + 'Detected '
        answer = answer + info

        for index, label in enumerate(labels):
            if index in unique_indices:
                if check_sf_features(test_features[index], label)[0] is False:
                    if language == 'chinese':
                        labels[index] = '未知'
                    else:
                        labels[index] = 'Unknown'
                else:
                    if language == 'chinese':
                        info = info + '{}，'.format(label)
                        answer = answer + '{}，'.format(label)
                    else:
                        info = info + '{},'.format(label)
                        answer = answer + '{},'.format(label)
            else:
                if language == 'chinese':
                    labels[index] = '未知'
                else:
                    labels[index] = 'Unknown'
        info = info[:-1]
        answer = answer[:-1]
        if language == 'english':
            info = info + ' in this image!'
            answer = answer + ' in this image!'
        else:
            info = info + '！'
            answer = answer + '！'

        if fr is not None:
            predict_info.append(info)
            fr.show_information(predict_info, predict=True)

        for index, label in enumerate(labels):
            face_position = face_boxes[index].astype(int)
            label_pixels, font_size, line_size, rect_size, up_offset = \
                None, None, None, None, None
            if face_number == 1:
                rect_size = 8
                if language == 'chinese':
                    label_pixels = 140 * len(label)
                    font_size = 140
                    up_offset = 140
                else:
                    label_pixels = 90 * len(label)
                    up_offset = 20
                    font_size = 5
                line_size = 9
            elif face_number < 4:
                rect_size = 7
                if language == 'chinese':
                    label_pixels = 140 * len(label)
                    font_size = 140
                    up_offset = 140
                else:
                    label_pixels = 70 * len(label)
                    up_offset = 20
                    font_size = 4
                line_size = 7
            elif face_number >= 4:
                rect_size = 6
                if language == 'chinese':
                    label_pixels = 100 * len(label)
                    font_size = 100
                    up_offset = 100
                else:
                    label_pixels = 50 * len(label)
                    up_offset = 20
                    font_size = 3
                line_size = 5

            dis = (label_pixels - (face_position[2]-face_position[0])) // 2

            if language == 'chinese':
                # The color coded storage order in cv2(BGR) and PIL(RGB) is different
                cv2img = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                pilimg = Image.fromarray(cv2img)
                # use PIL to display Chinese characters in images
                draw = ImageDraw.Draw(pilimg)
                font = ImageFont.truetype('../resources/STKAITI.ttf',
                                          font_size, encoding="utf-8")
                draw.text((face_position[0] - dis, face_position[1]-up_offset),
                          label, (0, 0, 255), font=font)
                # convert to cv2
                image_data = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
            else:
                cv2.putText(image_data, label,
                            (face_position[0]-dis, face_position[1]-up_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, font_size,
                            (255, 0, 0), line_size)
            cv2.rectangle(image_data, (face_position[0], face_position[1]),
                          (face_position[2], face_position[3]),
                          (0, 255, 0), rect_size)
        '''
        elif state == 'probability':
        best_probability_model = clf.load_best_probability_model(all_data=all_data)
        labels, probability = predict_probability(best_probability_model, test_features,
                                                  face_number, all_data=all_data)
        for i in range(face_number):
            answer = answer + '第{}张人脸的预测标签是{}，概率为{}.\n'.format(
                i + 1, labels[i], probability[i])
            info = '第{}张人脸的预测标签是{}，概率为{}.'.format(
                    i + 1, labels[i], probability[i])
            if fr is not None:
                predict_info.append(info)
                fr.show_information(predict_info, predict=True)
            if output:
                print(info)

            face_position = face_boxes[i].astype(int)
            cv2.rectangle(image_data, (face_position[0], face_position[1]), (
                face_position[2], face_position[3]), (0, 255, 0), 8)
            cv2.putText(image_data, labels[i], (face_position[0], face_position[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 6)

        '''
    elif state == 'search':
        best_probability_model = clf.load_best_probability_model(all_data=all_data)
        _, _, unique_labels, unique_probability, unique_indices = \
            find_people_from_image(best_probability_model, test_features, face_number, all_data=all_data)
        n = unique_labels.shape[0]

        found_indices = []
        if language == 'chinese':
            info = '从图像中找到'
        else:
            info = 'Found '
        answer = answer + info
        for i in range(n):
            if unique_labels[i] not in used_labels:
                continue
            index = unique_indices[i]
            if check_sf_features(test_features[index], unique_labels[i])[0] is False:
                continue
            '''
            answer += '{}th unique label is {} and the probability is {}.\n'.format(
                index, unique_labels[i], unique_probability[i])
            info = '{}th unique label is {} and the probability is {}.'.format(
                index, unique_labels[i], unique_probability[i])
            if output:
                print(unique_labels[i], unique_probability[i], index)
            '''
            if language == 'chinese':
                answer = answer + '{}，'.format(unique_labels[i])
                info = info + '{}，'.format(unique_labels[i])
            else:
                answer = answer + '{},'.format(unique_labels[i])
                info = info + '{},'.format(unique_labels[i])
            found_indices.append(i)

        info = info[:-1]
        answer = answer[:-1]
        if language == 'english':
            info = info + ' in this image!'
            answer = answer + ' in this image!'
        else:
            info = info + '！'
            answer = answer + '！'

        if fr is not None:
            predict_info.append(info)
            fr.show_information(predict_info, predict=True)

        for i in found_indices:
            index = unique_indices[i]
            face_position = face_boxes[index].astype(int)
            label_pixels, font_size, line_size, rect_size, up_offset = \
                None, None, None, None, None
            if face_number == 1:
                rect_size = 8
                if language == 'chinese':
                    label_pixels = 140 * len(unique_labels[i])
                    font_size = 140
                    up_offset = 140
                else:
                    label_pixels = 90 * len(unique_labels[i])
                    up_offset = 20
                    font_size = 5
                line_size = 9
            elif face_number < 4:
                rect_size = 7
                if language == 'chinese':
                    label_pixels = 140 * len(unique_labels[i])
                    font_size = 140
                    up_offset = 140
                else:
                    label_pixels = 70 * len(unique_labels[i])
                    up_offset = 20
                    font_size = 4
                line_size = 7
            elif face_number >= 4:
                rect_size = 6
                if language == 'chinese':
                    label_pixels = 100 * len(unique_labels[i])
                    font_size = 100
                    up_offset = 100
                else:
                    label_pixels = 50 * len(unique_labels[i])
                    up_offset = 20
                    font_size = 3
                line_size = 5

            dis = (label_pixels - (face_position[2]-face_position[0])) // 2

            if language == 'chinese':
                cv2img = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                pilimg = Image.fromarray(cv2img)
                draw = ImageDraw.Draw(pilimg)
                font = ImageFont.truetype('../resources/STKAITI.ttf',
                                          font_size, encoding="utf-8")
                draw.text((face_position[0] - dis, face_position[1]-up_offset),
                          unique_labels[i], (0, 0, 255), font=font)
                image_data = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
            else:
                cv2.putText(image_data, unique_labels[i],
                            (face_position[0] - dis, face_position[1]-up_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, font_size,
                            (255, 0, 0), line_size)
            cv2.rectangle(image_data, (face_position[0], face_position[1]),
                          (face_position[2], face_position[3]),
                          (0, 255, 0), rect_size)

    return answer, image_data

