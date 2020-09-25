import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from keras.layers import Input, Embedding, multiply, BatchNormalization
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score

SEED = 1337
np.random.seed(SEED)

randomDim = 100

def load_dataset():
    project_dir = Path(__file__).resolve().parents[2]
    data_processed = project_dir.joinpath('data/processed')
    files = ['X_train.csv', 'y_train.csv', 'X_val.csv', 'y_val.csv', 'X_test.csv', 'y_test.csv']
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(data_processed.joinpath(f).as_posix()))
    return dfs

def build_generator(latent_dim,data_dim):
    model = Sequential()

    model.add(Dense(16, input_dim=latent_dim))

    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(32, input_dim=latent_dim))

    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(data_dim,activation='tanh', name="gout1"))

    model.summary()

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)

def build_discriminator(data_dim,num_classes):
    model = Sequential()
    model.add(Dense(31,input_dim=data_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(0.25))
    model.add(Dense(16,input_dim=data_dim))
    model.add(LeakyReLU(alpha=0.2))
    
    model.summary()
    img = Input(shape=(data_dim,))
    features = model(img)
    valid = Dense(1, activation="sigmoid", name="dout1")(features)
    label = Dense(num_classes+1, activation="softmax", name="dout2")(features)
    return Model(img, [valid, label])

def get_gen_dim():
    generator = build_generator(latent_dim=10,data_dim=29)
    discriminator = build_discriminator(data_dim=29,num_classes=2)

    return generator, discriminator

def get_model(generator, discriminator):
    optimizer = Adam(0.0002, 0.5)
    discriminator.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
        loss_weights=[0.5, 0.5],
        optimizer=optimizer,
        metrics=['accuracy'])
    noise = Input(shape=(10,))
    img = generator(noise)
    discriminator.trainable = False
    valid,_ = discriminator(img)
    combined = Model(noise , valid)
    combined.compile(loss=['binary_crossentropy'],
        optimizer=optimizer)
    combined.summary()
    return combined



def train(X_train,y_train,
          X_test,y_test,
          generator,discriminator,
          combined,
          num_classes,
          epochs, 
          batch_size=128):
    
    f1_progress = []
    half_batch = int(batch_size / 2)

    noise_until = epochs

    # TODO: reintroduce class weights

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half batch of images
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = X_train[idx]

        # Sample noise and generate a half batch of new images
        noise = np.random.normal(0, 1, (half_batch, 10))
        gen_imgs = generator.predict(noise)

        valid = np.ones((half_batch, 1))
        fake = np.zeros((half_batch, 1))

        labels = to_categorical(y_train[idx], num_classes=num_classes+1)
        fake_labels = to_categorical(np.full((half_batch, 1), num_classes), num_classes=num_classes+1)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(imgs, [valid, labels])
        d_loss_fake = discriminator.train_on_batch(gen_imgs, [fake, fake_labels])
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


        # ---------------------
        #  Train Generator
        # ---------------------

        noise = np.random.normal(0, 1, (batch_size, 10))
        validity = np.ones((batch_size, 1))

        # Train the generator
        g_loss = combined.train_on_batch(noise, validity)

        # Plot the progress
        print ("%d [D loss: %f, acc: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss))
        
        if epoch % 10 == 0:
            _,y_pred = discriminator.predict(X_test,batch_size=batch_size)
            #print(y_pred.shape)
            y_pred = np.argmax(y_pred[:,:-1],axis=1)
            
            f1 = f1_score(y_test,y_pred)
            print('Epoch: {}, F1: {:.5f}, F1P: {}'.format(epoch,f1,len(f1_progress)))
            f1_progress.append(f1)
    project_dir = Path(__file__).resolve().parents[2]
    model_dir = project_dir.joinpath('models')
    discriminator.save(model_dir.joinpath('discriminator.h5').as_posix())
    generator.save(model_dir.joinpath('generator.h5').as_posix())
    combined.save(model_dir.joinpath('combined.h5').as_posix())
    return f1_progress

def plot_progress(f1_progress):
    fig = plt.figure(figsize=(10,7))
    plt.plot(f1_progress)
    plt.xlabel('10 Epochs')
    plt.ylabel('F1 Score Validation')