import tensorflow as tf
import nltk
import pandas as pd
import numpy as np
import cv2 as cv
from fastai.imports import *
import os, glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import timeit
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.constraints import Constraint

tf.gfile = tf.io.gfile
import tensorflow_hub as hub
from tensorflow import keras
import keras.utils as image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.models import load_model

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from keras.layers import Bidirectional,LSTM, Add,GRU,MaxPooling1D, GlobalMaxPool1D, GlobalMaxPooling1D, Dropout,Conv1D,Embedding,Flatten, Input, Layer,GlobalAveragePooling1D,Activation,Lambda,LayerNormalization, Concatenate, Average,AlphaDropout,Reshape, multiply

import contractions
from bs4 import BeautifulSoup
from keras.utils import to_categorical
from sklearn import preprocessing
#from keras.preprocessing.text import Tokenizer
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer,PorterStemmer
from tensorflow.keras.layers import TextVectorization
import tqdm
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import roc_curve, auc
import spacy
from scipy import stats
from spacy import displacy
nlp = spacy.load("en_core_web_sm")
#import bert_tokenizer as tok
import absl.logging
import tensorflow_hub as hub
from bert import tokenization
#absl.logging.set_verbosity(absl.logging.ERROR)

print('DEBUT..................DEFINITION DES FONCTIONS')
m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(m_url, trainable=True)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

##############IMPORTATION DES DATASETS#############################

dataTest = pd.read_csv('tweets.txt', sep="\t",header=0, engine='python',encoding= 'raw_unicode_escape',quoting=3)
dataOtre = pd.read_csv('DevSetOtre.txt', sep="\t",header=0, engine='python',encoding= 'raw_unicode_escape')
dataTest = dataTest.query('label == "fake" or label == "real"')
dataOtre = dataOtre.query('label == "fake" or label == "real"')
dataOtre = dataOtre.reset_index(drop=True)
dataTest = dataTest.reset_index(drop=True)

##############LECTURE DES IMAGE ###############"
dataImageDev = pd.DataFrame(columns=['nomImage','image','label'])
dataImageTest = pd.DataFrame(columns=['nomImage','image','label'])
textListe = []
imageListe = []
labelImage = []
labelText = []
#Former un dataframe les images et le label
i=0
j=0
repDev=glob.glob('MediaEval2016/DevImages/*')
#repTest=glob.glob('MediaEval2016/TestImages/*')
while i < len(repDev):
  try:
    #img = Image.open(filename)
    img2 = cv.imread(repDev[i])
    imgResize = cv.resize(img2, (224,224))
    #print("AAAAA ",np.array(img))
    chemin= repDev[i].split("/")
    imgName = chemin[len(chemin)-1]
    imgName=imgName.replace(" ", "")
    nb = len(imgName)-4
    dataImageDev.loc[i] = [imgName[:nb],imgResize,0]
  except:
    pass
  i=i+1

#####################REGROUPEMENT DES IMAGES ET TWEETS DANS UN MEME DATAFRAME#############################
dataImageTextDev = pd.DataFrame(columns=['text','nomImage','image','label'])
dataImageTextTest = pd.DataFrame(columns=['text','nomImage','image','label'])
k=0
j=0
imageListe=[]
labelImage=[]
textListe=[]
k=0
sortie =0
sansImage =[]
i=0
while i <len(dataOtre):
  #name=dataOtre.loc[i,'imageId(s)']
  name = dataOtre['imageId(s)'][i]
  trouver=0
  for index, valeur in dataImageDev['nomImage'].items():
    if(name == valeur):
      trouver=1
      tweetClean = dataOtre['tweetText'][i]
      dataImageTextDev.loc[k]=[str(tweetClean),name,dataImageDev['image'][index],dataOtre['label'][i]]
      k=k+1
  i=i+1
i=0
j=0
k=0
################################
i=0
while i <len(dataTest):
  #name=dataOtre.loc[i,'imageId(s)']
  name = dataTest['imageId(s)'][i]
  trouver=0
  for index, valeur in dataImageDev['nomImage'].items():
    if(name == valeur):
      trouver=1
      tweetClean = dataTest['tweetText'][i]
      dataImageTextTest.loc[k]=[str(tweetClean),name,dataImageDev['image'][index],dataTest['label'][i]]
      k=k+1
  i=i+1

################RESUME DU MODEL##########################
docsClean = []
model = fake_virtual(bert_layer, max_len=max_len)
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

################### ENTRAINEMENT ET VALIDATION CROISEE##################################
i=0
kf = KFold(n_splits=5, shuffle=True)
print('DEBUT FORMATION DU MODEL........')
save_dir = '/saved_models/'
fold_var = 1
max_len=16
result = []
scores_loss = []
scores_acc = []
scores_pre = []
scores_rap = []
fold_var = 1
for train_indices, val_indices in kf.split(dataImageTextDev):
    train = dataImageTextDev.iloc[train_indices]
    val = dataImageTextDev.iloc[val_indices]
    label = preprocessing.LabelEncoder()

    trainText = bert_encode(train['text'], tokenizer, max_len=max_len)
    trainImage = train['image']
    trainImage = trainImage.to_numpy()
    trainImage = np.array([val for val in trainImage])
    trainLabel = label.fit_transform(train['label'])
    trainLabel = to_categorical(trainLabel)
    labels = label.classes_
    valText = bert_encode(val['text'], tokenizer, max_len=max_len)
    valImage = val['image']
    valImage = valImage.to_numpy()
    valImage = np.array([val for val in valImage])
    valLabel = label.fit_transform(val['label'])
    valLabel = to_categorical(valLabel)
    valLabel = valLabel
    testText = bert_encode(dataImageTextTest['text'], tokenizer, max_len=max_len)
    testImage = dataImageTextTest['image']
    testImage = testImage.to_numpy()
    testImage = np.array([val for val in testImage])
    testLabel = label.fit_transform(dataImageTextTest['label'])
    testLabel = to_categorical(testLabel)
    testLabel = testLabel

    print(fold_var)
    file_path = save_dir+'model_'+str(fold_var)+".hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='loss', verbose=0, save_best_only=True, mode='min')
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor="loss", mode="min", patience=8)
    callbacks_list = [checkpoint,earlystopping]
    hist = model.fit(x=[trainText,trainImage],y=trainLabel,epochs=30,batch_size=70,validation_data=([valText, valImage], valLabel), callbacks=callbacks_list, verbose=1)
    model.load_weights(file_path)
    score = model.evaluate([valText, valImage],valLabel, verbose=0)
    scores_loss.append(score[0])
    scores_acc.append(score[1])
    scores_pre.append(score[2])

  ########################
    scores_rap.append(score[3])
    result.append(model.predict([testText,testImage]))
    tf.keras.backend.clear_session()
    fold_var += 1
#############################
value_min = min(scores_loss)
value_index = scores_loss.index(value_min)
model.load_weights(save_dir+'model_'+str(value_index)+".hdf5")
best_model = model
best_model.save_weights('model_weights.h5')
best_model.save('model_keras.h5')

##############################
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
epochs = range(len(train_acc))
plot_accuracy(epochs, train_acc,val_acc)
plot_loss(epochs, train_loss,val_loss)

###################MATRICE DE CONFUSION################
test_loss = best_model.evaluate([testText,testImage], testLabel, verbose=1)
results = best_model.predict([testText,testImage])
print(results)
pred_labels = np.argmax(results, axis = 1)
print(pred_labels)
print(testLabel)
cm = confusion_matrix(testLabel, pred_labels)
class_names = ['fake','real']
#Transform to df for easier plotting
cm_df = pd.DataFrame(cm)
final_cm = cm_df
plt.figure(figsize = (5,5))
sns.heatmap(final_cm, annot = True,cmap="YlGnBu",cbar=False,fmt='d')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual class')
plt.xlabel('Prediction class')
plt.show()

###################ROC CURVE AND AUC#########################
from sklearn.metrics import roc_curve,auc
from itertools import cycle
y_pred = best_model.predict([testText,testImage])
y_pred_ravel = y_pred.ravel()
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(new_class):
    fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_pred[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])
#colors = cycle(['red', 'green','black'])
colors = cycle(['red', 'green','black','blue', 'yellow','purple','orange'])
for i, color in zip(range(new_class), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0}'''.format(final_label[i]))
