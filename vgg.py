import mtcnn
from keras_vggface.vggface import VGGFace
import matplotlib.pyplot as plt
from keras_vggface import utils
from PIL import Image
import numpy as np
def extract_face(filename, size=(224,224)):
    pixels = plt.imread(filename)
    detector = mtcnn.MTCNN()
    results = detector.detect_faces(pixels)
    x1,y1,width,height = results[0]['box']
    x2,y2 = x1+width, y1+height
    face = pixels[y1:y2,x1:x2]
    image = Image.fromarray(face)
    image = image.resize(size)
    return np.asarray(image)

pixles = extract_face('sharon_stone1.jpg')
model = VGGFace(model='resnet50')

pixles = pixles.astype('float32')

samples = np.expand_dims(pixles,axis=0)
samples = utils.preprocess_input(samples,version=2)

yhat= model.predict(samples)

results = utils.decode_predictions(yhat)

for r in results[0]:
    print('%s: %.3f%%' % (r[0],r[1]*100))