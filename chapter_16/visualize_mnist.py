# Visualize the MNIST dataset
import numpy as np
import matplotlib.pyplot as plt

# Load test data from local file data/t10k-images.idx3-ubyte
mnist_test_data = open('data/t10k-images.idx3-ubyte', 'rb').read()
mnist_test_data = np.frombuffer(mnist_test_data, dtype=np.uint8, offset=16)
mnist_test_data = mnist_test_data.reshape(-1, 28, 28)

# Load test labels from local file data/t10k-labels.idx1-ubyte
mnist_test_labels = open('data/t10k-labels.idx1-ubyte', 'rb').read()
mnist_test_labels = np.frombuffer(mnist_test_labels, dtype=np.uint8, offset=8)

# Load train data from local file data/train-images.idx3-ubyte
mnist_train_data = open('data/train-images.idx3-ubyte', 'rb').read()
mnist_train_data = np.frombuffer(mnist_train_data, dtype=np.uint8, offset=16)
mnist_train_data = mnist_train_data.reshape(-1, 28, 28)

# Load train labels from local file data/train-labels.idx1-ubyte
mnist_train_labels = open('data/train-labels.idx1-ubyte', 'rb').read()
mnist_train_labels = np.frombuffer(mnist_train_labels, dtype=np.uint8, offset=8)

# Print the shape of the data and labels
print(mnist_test_data.shape)
print(mnist_test_labels.shape)

# VIsualize the first 10 images of test and train data with labels
fig, ax = plt.subplots(2, 10, figsize=(10, 1))
for i in range(10):
    ax[0, i].imshow(mnist_test_data[i], cmap='gray')
    ax[0, i].axis('off')
    ax[0, i].set_title(mnist_test_labels[i])
    ax[1, i].imshow(mnist_train_data[i], cmap='gray')
    ax[1, i].axis('off')
    ax[1, i].set_title(mnist_train_labels[i])

plt.show()
