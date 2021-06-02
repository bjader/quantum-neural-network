import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense  # Dense layers are "fully connected" layers
from keras.models import Sequential  # Documentation: https://keras.io/models/sequential/

# Load training and test data
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Remove all data which isn't labelled as a 3 or 6
train_mask = np.isin(y_train, [3, 6])
test_mask = np.isin(y_test, [3, 6])

x_train, y_train = x_train[train_mask], y_train[train_mask]
x_test, y_test = x_test[test_mask], y_test[test_mask]

print("Training data shape: ", x_train.shape)
print("Test data shape", x_test.shape)

dsize = 4

# Downsample image data
x_train = np.array([cv2.resize(sample, dsize=(dsize, dsize), interpolation=cv2.INTER_CUBIC) for sample in x_train])
x_test = np.array([cv2.resize(sample, dsize=(dsize, dsize), interpolation=cv2.INTER_CUBIC) for sample in x_test])

# Flatten the images
image_vector_size = dsize ** 2
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)

print("Training data shape: ", x_train.shape)  # (12049, 784) -- 12049 images, each 784x1 vector
print("Test data shape", x_test.shape)  # (1968, 784) -- 1968 images, each 784x1 vector

# Convert from greyscale to black and white binary
_, x_train = cv2.threshold(x_train, 127, 255, cv2.THRESH_BINARY)
_, x_test = cv2.threshold(x_test, 127, 255, cv2.THRESH_BINARY)

# Map the labels "3" and "6" to labels starting from 0 e.g. ("0" and "1")
y_train = np.array([0 if val == 3 else 1 for val in y_train])
y_test = np.array([0 if val == 3 else 1 for val in y_test])

# Convert to "one-hot" vectors using the to_categorical function
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create the network
num_classes = 2  # two unique digits

model = Sequential()
# The input layer requires the special input_shape parameter which should match
# the shape of our training data.
model.add(Dense(units=3, activation='sigmoid', input_shape=(image_vector_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.summary()

model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 50
history = model.fit(x_train, y_train, batch_size=128, epochs=epochs, verbose=False, validation_split=.1)
loss, accuracy = model.evaluate(x_test, y_test, verbose=True)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.ylim(0.5, 1.0)
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best', title=f'Test accuracy: {accuracy:.3}')
plt.show()

print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')
