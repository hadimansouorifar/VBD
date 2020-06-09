from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(1568,))
encoded = Dense(256, activation='relu')(input_img)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)

decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(1568, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


from keras.datasets import fashion_mnist
import numpy as np
(x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)


xtr=[]
xte=[]
for i in range(0,len(x_train)):
   if (y_train[i]==0):
      xtr.append(x_train[i])


for i in range(0,len(x_test)):
   if (y_test[i]==0):
      xte.append(x_test[i])

print(len(xtr))
xtr = np.asarray(xtr)
xte = np.asarray(xte)

xtr2 = np.zeros(((980*980),1568), dtype=np.float)
xte2 = np.zeros(((980*980),1568), dtype=np.float)
c=0
for i in range(0,980):
            for p in range(0,980):

                if (i!=p):
                    t1 = np.concatenate((xtr[i], xtr[p]), axis=None)
                    t2 = np.concatenate((xte[i], xte[p]), axis=None)


                    xtr2[c] = t1
                    xte2[c] = t2

                    c=c+1


autoencoder.fit(xtr2, xtr2,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(xte2, xte2))



# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = autoencoder.predict(xte)
decoded_imgs = autoencoder.predict(encoded_imgs)
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()