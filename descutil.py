#!/usr/bin/env python 3
# -*- coding : utf-8 -*-

from math import sqrt

import cv2
import dlib
import numpy as np

# Path declaration
PRED_PATH = 'model/shape_predictor_68_face_landmarks.dat'
RECG_PATH = 'model/dlib_face_recognition_resnet_model_v1.dat'

# Models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PRED_PATH)
rec_model = dlib.face_recognition_model_v1(RECG_PATH)


def gen_descriptor(org, fn):
    b, g, r = cv2.split(org)
    img = cv2.merge((r, g, b))

    face = detector(img, 1)[0]  # Select the first face detected as object
    shape = predictor(img, face)
    for i, pt in enumerate(shape.parts()):
        pt_pos = (pt.x, pt.y)
        cv2.circle(org, pt_pos, 3, (120, 0, 200), 3)
    descriptor = rec_model.compute_face_descriptor(img, shape)
    cv2.imwrite(f'output/parsed_{fn}.jpg', org)
    # print(f'output/parsed_{fn}.jpg')
    return np.array(descriptor).reshape((1, 128))


euclidean = lambda veca, vecb: sqrt(np.sum(np.square(veca - vecb)))

if __name__ == '__main__':
    raise SystemExit("This is the descutil module, not the main program. Run `python3 main.py`.")
