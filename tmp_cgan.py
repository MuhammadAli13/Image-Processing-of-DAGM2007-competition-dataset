#!/usr/bin/env python
# coding: utf-8

# In[1]:


# example of training an conditional gan on the fashion mnist dataset
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.utils import plot_model
plot_model.__defaults__='model.png', True, True, 'TB', False, 96
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm_notebook
def c(img):
    plt.imshow(img.reshape(96,96),cmap='gray')
def random_6_plot(g_model):
    [z_input,labels_input]=generate_latent_points(latent_dim,6)
    labels_input = np.array([0,0,0,1,1,1])
    preds = g_model.predict([z_input,labels_input])
    plt.figure(figsize=(8,5))
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(preds[i].reshape(96,96),cmap='gray')
    plt.show()


# # Data Processing

# In[2]:


bad_images_Main=np.array([cv2.resize(cv2.imread(i,0),(96,96)) for i in glob('Class4_def/*.png')])
good_images_Main=np.array([cv2.resize(cv2.imread(i,0),(96,96)) for i in glob('Class4/*.png')])


# In[3]:


together_imgs = np.concatenate([good_images_Main,bad_images_Main])
labels=np.concatenate([np.ones(len(good_images_Main)),np.zeros(len(bad_images_Main))])
idx=np.arange(1150)
np.random.shuffle(idx)
trainX=together_imgs[idx]
trainy=labels[idx]
X = expand_dims(trainX, axis=-1)
# convert from ints to floats
X = X.astype('float32')
# scale from [0,255] to [-1,1]
X = (X - 127.5) / 127.5
dataset = X,trainy


# # Functions

# In[4]:


def generate_real_samples(dataset, n_samples):
	# split into images and labels
	images, labels = dataset
	# choose random instances
	ix = randint(0, images.shape[0], n_samples)
	# select images and labels
	X, labels = images[ix], labels[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return [X, labels], y
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=2):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = zeros((n_samples, 1))
	return [images, labels_input], y


# In[5]:


def define_discriminator(in_shape=(96,96,1), n_classes=2):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# scale up to image dimensions with linear activation
	n_nodes = in_shape[0] * in_shape[1]
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((in_shape[0], in_shape[1], 1))(li)
	# image input
	in_image = Input(shape=in_shape)
	# concat label as a channel
	merge = Concatenate()([in_image, li])
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
	fe = LeakyReLU(alpha=0.2)(fe)

    
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
    
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
    
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# output
	out_layer = Dense(1, activation='sigmoid')(fe)
	# define model
	model = Model([in_image, in_label], out_layer)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
plot_model(define_discriminator())


# In[6]:


def define_generator(latent_dim, n_classes=2):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# linear multiplication
	n_nodes = 12 * 12
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((12, 12, 1))(li)
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 128 * 12 * 12
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((12, 12, 128))(gen)
	# merge image gen and label input
	merge = Concatenate()([gen, li])
	# upsample to 14x14
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
    
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
    
	# output
	out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
	# define model
	model = Model([in_lat, in_label], out_layer)
	return model
plot_model(define_generator(latent_dim=100))


# In[ ]:





# In[7]:


latent_dim=100
d_model = define_discriminator()
g_model = define_generator(latent_dim=100)
noise = Input(shape=(latent_dim,))
label = Input(shape=(1,))
img = g_model([noise, label])
d_model.trainable = False
validity = d_model([img, label])
gan_model = Model(input=[noise, label], output=validity)
gan_model.compile(loss=['binary_crossentropy'],
                                optimizer=Adam(lr=0.0002, beta_1=0.5),
                                metrics=['accuracy'])


# In[8]:


n_epochs=100
n_batch=128
bat_per_epo = int(dataset[0].shape[0] / n_batch)
half_batch = int(n_batch / 2)
iteration = n_epochs*bat_per_epo
iteration


# In[9]:


for i in tqdm_notebook(range(iteration)):
    # get randomly selected 'real' samples
    [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
    # update discriminator model weights
    d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
    # generate 'fake' examples
    [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
    # update discriminator model weights
    d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
    # prepare points in latent space as input for the generator
    [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
    # create inverted labels for the fake samples
    y_gan = ones((n_batch, 1))
    # update the generator via the discriminator's error
    g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
    g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
    g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
    if i%5==0:
        print("epoch ",i/bat_per_epo,'d_loss1 ',d_loss1,'d_loss2 ',d_loss2
             ,'g_loss ',g_loss)
        random_6_plot(g_model)


# In[ ]:





# In[ ]:





# In[10]:


import tensorflow as tf
with tf.Session() as sess:
    devices = sess.list_devices()


# In[11]:


devices


# In[12]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[13]:


from keras.models import save_model


# In[14]:


save_model(g_model,'g_model.hdf5')


# In[15]:


save_model(d_model,'d_model.hdf5')


# In[17]:


save_model(gan_model,'gan_model.hdf5')


# In[ ]:
