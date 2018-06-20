# -*- coding: utf-8 -*-

from scipy import misc
from skimage import io
import dlib
import cv2
import os
import tensorflow as tf
import numpy as np
from face_recognition import align_dlib, detect_face, facenet


def cv_imread(image_path, color_mode='rgb'):
    """ Uses opencv to read the image path which contains Chinese.

    Inputs:
      - image_path: A string, path of the image.
      - color_mode: 'rgb' or 'grayscale'.

    Return:
      - image_data: A numpy array, data of the image.
    """
    if color_mode not in {'rgb', 'grayscale'}:
        raise ValueError('color_mode must be \'rgb\' or \'grayscale\'.')

    if color_mode == 'rgb':
        image_data = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),
                                  cv2.IMREAD_COLOR)
    else:
        image_data = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),
                                  cv2.IMREAD_GRAYSCALE)
    return image_data


def process(state, path, fr=None, image_size=144, output=True,
            use_alignment=True, language='chinese'):
    """ Image processing.
    Inputs:
      - state: 'training' or 'test'.
      - image_size: The image size of the MTCNN network.
      - use_alignment: True or False, whether to use face alignment.
      - output: True or False, whether to output training process.
      - language: 'chinese' or 'english'.

    Returns:
      when state is 'train':
          - features: A numpy array with shape of (N, D) contains the
                features of all training images.
          - labels: A numpy array with shape of (N, ) contains the
                labels of all training images.
      when state is 'test':
          - test_features: A numpy array with shape of (N, D) contains
                the features of the test images.
          - face_number: An integer, the number of faces we found in
                the test image.
          - face_boxes: A list, contains all position of the faces.
    """
    if state not in {'training', 'test'}:
        raise ValueError('{} is not a valid argument!'.format(state))

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    gpu_memory_fraction = 0.6

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                log_device_placement=False))
        with sess.as_default():
            if os.path.exists('models'):
                p_net, r_net, o_net = detect_face.create_mtcnn(sess,
                                                               'models/mtcnn/')
            else:
                p_net, r_net, o_net = detect_face.create_mtcnn(sess,
                                                               '../models/mtcnn/')

    if os.path.exists('models'):
        predictor_model = 'models/shape_predictor_68_face_landmarks.dat'
    else:
        predictor_model = '../models/shape_predictor_68_face_landmarks.dat'

    # "models/shape_predictor_68_face_landmarks.dat"
    # face_pose_predictor = dlib.shape_predictor(predictor_model)
    face_aligner = align_dlib.AlignDlib(predictor_model)

    if os.path.exists('models'):
        model_dir = 'models/20170512-110547/20170512-110547.pb'  # model directory
    else:
        model_dir = '../models/20170512-110547/20170512-110547.pb'  # model directory
    tf.Graph().as_default()
    sess = tf.Session()
    facenet.load_model(model_dir)
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    if output:
        process_info = []
    if state == 'training':
        root_dir = path
        labels = []  # the label of all faces.
        features = []  # the features of all faces.

        for image_dir in os.listdir(root_dir):
            dir_path = os.path.join(root_dir, image_dir)

            if os.path.isdir(dir_path) is False:
                raise ValueError('{} is not a directory!'.format(dir_path))

            now_image_label = image_dir
            now_image_list = os.listdir(dir_path)
            # now_image_number = len(now_image_list)

            special_features = []
            for index, image_name in enumerate(now_image_list):
                if image_name[-4:] in ['.jpg', '.png', '.JPG', '.PNG']:
                    now_image_path = os.path.join(dir_path, image_name)
                    if output:
                        if language == 'chinese':
                            info = '检测图像{}，标签是{}。'.format(now_image_path,
                                                          now_image_label)
                        else:
                            info = 'Detect image {}, label is {}.'.format(now_image_path,
                                                                          now_image_label)
                        if fr is not None:
                            process_info.append(info)
                            fr.show_information(process_info, pro_output=True)
                            # fr.show_information(info, pro_output=True)
                        print(info)
                    image_data = misc.imread(now_image_path)
                    # image_data = io.imread(now_image_path)
                    boxes, _ = detect_face.detect_face(
                        image_data, minsize, p_net, r_net, o_net, threshold, factor)

                    face_number = boxes.shape[0]
                    if output:
                        if face_number in {0, 1}:
                            if language == 'chinese':
                                info = '从这张图片中发现{}张人脸。'.format(face_number)
                            else:
                                info = 'Found {} face in this image.'.format(face_number)
                            if fr is not None:
                                process_info.append(info)
                                fr.show_information(process_info, pro_output=True)
                                # fr.show_information(info, pro_output=True)
                            print(info)
                        else:
                            if language == 'chinese':
                                info = '从这张图片中发现{}张人脸。'.format(face_number)
                            else:
                                info = 'Found {} faces in this image.'.format(face_number)
                            if fr is not None:
                                process_info.append(info)
                                fr.show_information(process_info, pro_output=True)
                                # fr.show_information(info, pro_output=True)
                            print(info)

                    # if face_number is not 1:
                    #    if output:
                    #        print('The number of faces is not legal!')
                    #    continue
                    # crop_data = None
                    for face_position in boxes:
                        face_position = face_position.astype(int)
                        # last_data = None
                        if use_alignment:
                            face_rect = dlib.rectangle(int(face_position[0]), int(
                                face_position[1]), int(face_position[2]), int(face_position[3]))

                            # pose_landmarks = face_pose_predictor(image_data, face_rect)
                            # image_landmarks = pose_landmarks.parts()

                            aligned_data = face_aligner.align(
                                image_size, image_data, face_rect,
                                landmarkIndices=align_dlib.AlignDlib.OUTER_EYES_AND_NOSE)
                            last_data = aligned_data
                        else:
                            crop_data = image_data[face_position[1]:face_position[3],
                                                   face_position[0]:face_position[2],
                                                   :]
                            last_data = cv2.resize(crop_data, (image_size, image_size))

                        # print(crop_data.shape) -> (crop_size, crop_size, 3)
                        # crop = image_data[face_position[1]:face_position[3], face_position[0]:face_position[2], :]
                        # crop = misc.imresize(crop, (crop_size, crop_size), interp='bilinear')
                        # crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
                        # crop = facenet.prewhiten(crop)

                        last_data = facenet.prewhiten(last_data)
                        labels.append(now_image_label)
                        if output:
                            if language == 'chinese':
                                info = '现在提取特征...'
                            else:
                                info = 'Now extracting features..'
                            if fr is not None:
                                process_info.append(info)
                                fr.show_information(process_info, pro_output=True)
                                # fr.show_information(info, pro_output=True)
                            print(info)
                        last_data = last_data.reshape((1, image_size, image_size, 3))
                        features.append(sess.run(embeddings, feed_dict={
                            images_placeholder: last_data, phase_train_placeholder: False})[0])
                        if face_number is 1:
                            special_features.append(features[-1])
                        if output:
                            info = 'OK!'
                            if fr is not None:
                                process_info.append(info)
                                fr.show_information(process_info, pro_output=True)
                                # fr.show_information(info, pro_output=True)
                            print(info)
            else:
                special_features = np.array(special_features)
                sf_num = special_features.shape[0]
                if sf_num > 5:
                    dis_sum = []
                    for sf in special_features:
                        dis_sum.append(np.sum((sf-special_features)**2))
                    indices = np.argsort(dis_sum)[:5]
                    special_features = special_features[indices]
                if os.path.exists('features'):
                    np.save(open('features/{}.npy'.format(now_image_label), 'wb'),
                            special_features)
                else:
                    np.save(open('../features/{}.npy'.format(now_image_label), 'wb'),
                            special_features)

        else:
            pass

        features = np.array(features)
        labels = np.array(labels)
        if language == 'chinese':
            info = '训练结束！'
        else:
            info = 'Training over!'
        if fr is not None:
            process_info.append(info)
            fr.show_information(process_info, pro_output=True)
            # fr.show_information(info, pro_output=True)
        print(info)
        return features, labels

    elif state == 'test':
        test_image_data = io.imread(path)
        test_features = []
        face_boxes, _ = detect_face.detect_face(
            test_image_data, minsize, p_net, r_net, o_net, threshold, factor)
        face_number = face_boxes.shape[0]

        if face_number is 0:
            return None, face_number, None
        else:
            index = 1

            for face_position in face_boxes:
                face_position = face_position.astype(int)
                face_rect = dlib.rectangle(int(face_position[0]), int(
                    face_position[1]), int(face_position[2]), int(face_position[3]))

                # test_pose_landmarks = face_pose_predictor(test_image_data, face_rect)
                # test_image_landmarks = test_pose_landmarks.parts()

                aligned_data = face_aligner.align(
                    image_size, test_image_data, face_rect, landmarkIndices=align_dlib.AlignDlib.OUTER_EYES_AND_NOSE)

                # plt.subplot(face_number, 1, index)
                # plt.imshow(aligned_data)
                # plt.axis('off')
                # plt.show()
                # cv2.imwrite('datasets/team_aligned/{}.jpg'.format(str(index)),
                #             cv2.cvtColor(aligned_data, cv2.COLOR_RGB2BGR))

                aligned_data = facenet.prewhiten(aligned_data)
                index += 1
                last_data = aligned_data.reshape((1, image_size, image_size, 3))
                test_features.append(sess.run(embeddings, feed_dict={
                    images_placeholder: last_data, phase_train_placeholder: False})[0])
            test_features = np.array(test_features)

            return test_features, face_number, face_boxes

