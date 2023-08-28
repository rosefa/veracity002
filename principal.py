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
