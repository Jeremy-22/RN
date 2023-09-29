import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

learning_rate = 0.001
epochs = 30
batch_size = 120

dataset=mnist.load_data()

dat=np.array(dataset)
print(dat[1,1].shape)
(x_train, y_train), (x_test, y_test) = dataset


plt.imshow(dat[0,0][10000])