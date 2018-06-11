import numpy as np
import cv2
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.models import Sequential 
from extract_bottleneck_features import *
import os
from keras.backend import clear_session

namefilename = os.path.join(os.getcwd(),"dognames.txt")
namefile = open(namefilename,'r')

dog_names = list()
for line in namefile:
	dog_names.append(line.rstrip())

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

#returns True if face is detected in image stored at imgpath
def face_detector(imgpath):
	img = cv2.imread(imgpath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray)
	#clear_session()
	return len(faces) > 0


ResNet50_model = ResNet50(weights='imagenet')
g = tf.get_default_graph()

	
def path_to_tensor(imgpath):
	img = image.load_img(imgpath, target_size=(224,224)) #loads RGB image as PIL.Image.Image type
	x = image.img_to_array(img) #converts PIL.Image.Image type to 3D tensor with shape (224,224,3)
	return np.expand_dims(x,axis=0) #converts 3D tensor to 4D tensor with shape (1,224,224,3) and returns it

def paths_to_tensor(imgpaths):
	tensorlist = [path_to_tensor(imgpath) for imgpath in imgpaths]
	return np.vstack(tensorlist)

def ResNet50_predict_labels(imgpath):
	img = preprocess_input(path_to_tensor(imgpath))
	return np.argmax(ResNet50_model.predict(img))

def dog_detector(imgpath):
	with g.as_default():
		prediction = ResNet50_predict_labels(imgpath)
		#clear_session()
		return ((prediction <= 268) & (prediction >= 151))


bottleneck_features = np.load('bottleneck_features/DogVGG19Data.npz')
train_VGG19 = bottleneck_features['train']
valid_VGG19 = bottleneck_features['valid']
test_VGG19 = bottleneck_features['test']

vgg19model = Sequential()
vgg19model.add(GlobalAveragePooling2D(input_shape=train_VGG19.shape[1:]))
vgg19model.add(Dense(512, activation='relu'))
vgg19model.add(Dropout(0.4))
vgg19model.add(Dense(133,activation='softmax'))

vgg19model.load_weights('saved_models/weights.best.VGG19.hdf5')

def predict_breed_vgg19(imgpath):
	bottleneck = extract_VGG19(path_to_tensor(imgpath))
	#clear_session()
	prediction = vgg19model.predict(bottleneck)
	#clear_session()
	return dog_names[np.argmax(prediction)]

def dog_breed_detector(imgpath):
	with g.as_default():
		breed = predict_breed_vgg19(imgpath)

	if dog_detector(imgpath):
		return "This dog is a {}".format(breed)
	elif face_detector(imgpath):
		return "This person looks like a {}".format(breed)
	else:
		return "Error. Can't tell if this is a human or dog. Please upload a clearer picture"

#print(dog_breed_detector("11136617_10205016343262737_1711585873368655203_n.jpg"))

