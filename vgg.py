import mtcnn
from keras_vggface.vggface import VGGFace
import matplotlib.pyplot as plt
from keras_vggface import utils
from PIL import Image
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine
import numpy as np
import os
import cv2
from glob import glob
from random import shuffle


def im2single(im):
    im = im.astype(np.float32) / 255
    return im


def extract_face(filename, size=(224, 224)):
    print(filename)
    pixels = plt.imread(filename)
    detector = mtcnn.MTCNN()
    results = detector.detect_faces(pixels)
    if len(results) == 0:
        image = cv2.resize(pixels, size)
        return np.asarray(image)
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1+width, y1+height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(size)
    return np.asarray(image)


def generate_embeddings(files):
    faces = []
    for f in files:
        faces.append(extract_face(f))
    samples = np.asarray(faces, np.float32)
    samples = utils.preprocess_input(samples, version=2)
    model = VGGFace(include_top=False, model='resnet50',
                    input_shape=(224, 224, 3))
    y = model.predict(samples)
    return y


def is_match(embd_1, embd_2, theta=0.5):
    score = cosine(embd_1, embd_2)
    score = 1-score
    # print('Cosine similarity score %.5f' % score)
    if score > theta:
        return True, score
    else:
        return False, score


def face_identification(filename):
    pixles = extract_face(filename)
    pixles = pixles.astype('float32')
    samples = np.expand_dims(pixles, axis=0)
    samples = utils.preprocess_input(samples, version=2)

    model = VGGFace(model='resnet50')
    yhat = model.predict(samples)

    results = utils.decode_predictions(yhat)

    for r in results[0]:
        # return r[0] # top match
        print('%s: %.3f%%' % (r[0], r[1]*100))


def face_verification(file1, file2):
    embeddings = generate_embeddings([file1, file2])
    if is_match(embd_1=embeddings[0], embd_2=embeddings[1])[0]:
        print('Same person')
    else:
        print('Not same person')


def get_cls():
    cls = []
    for root, dirs, files in os.walk('./test'):
        for d in dirs:
            cls.append(d)
    return cls


def get_img_paths(cls, type='test'):
    train_image_paths = []
    test_image_paths = []
    train_labels = []
    test_labels = []
    image_paths = []
    image_labels = []
    for c in cls:
        pth = os.path.join(type, c, '*.{:s}'.format('jpg'))
        pth = glob(pth)
        if type == 'veri_test':
            image_paths.extend(pth)
            image_labels.extend([c]*len(pth))
        else:
            shuffle(pth)
            test_pth = pth[:20]
            train_pth = pth[20:80]
            train_image_paths.extend(train_pth)
            train_labels.extend([c]*len(train_pth))
            test_image_paths.extend(test_pth)
            test_labels.extend([c]*len(test_pth))

    if type == 'veri_test':
        return image_paths, image_labels
    return train_image_paths, test_image_paths, train_labels, test_labels


def svm_classify(train_feats, train_labels, test_image_feats):

    categories = list(set(train_labels))
    test_labels = []
    test_conf = []
    # construct 1 vs all SVMs for each category
    svms = {cat: SVC(kernel='linear',C=1)
            for cat in categories}

    for cat, svm in svms.items():
        y = [1 if i == cat else 0 for i in train_labels]
        svm.fit(train_feats, y)

    for t in test_image_feats:
        confidences = []

        for cat, svm in svms.items():  # calculate confidences for every svm
            w = svm.coef_  # W*X + B
            b = svm.intercept_
            wx = np.dot(w, t)
            conf = float(wx+b)
            confidences.append(conf)
        sorted_conf = np.argsort(confidences)  # min to max
        best_match = categories[sorted_conf[-1]]
        best_conf = confidences[sorted_conf[-1]]
        test_labels.append(best_match)
        test_conf.append(best_conf)

    return test_labels, test_conf


if __name__ == "__main__":
    # face_verification('zzn_1.jpg', 'ajw_1.jpg')
    face_identification('sharonstone_1.jpg')
    pass
