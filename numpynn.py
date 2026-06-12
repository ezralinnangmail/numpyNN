import numpy as np

np.random.seed(5)

import os
from PIL import Image

import time


# image loading pipeline
def read_images(folder_path):
    directory_path = folder_path
    images = []
    folder_labels = []

    for folder in os.listdir(directory_path):
        for file in os.listdir(directory_path + f"/{folder}/"):
            image_path = directory_path + f"/{folder}/" + file
            image_array = np.array(Image.open(image_path))
            images.append(image_array)
            folder_labels.append(int(folder))

    all_img_array = np.array(images)
    all_labels_array = np.array(folder_labels)

    return all_img_array, all_labels_array


# normalize inputs for faster and more reliable gradient descent
def normalize(arr, mean, std):
    return (arr - mean) / (std + 1e-7)


# turns N 2D images into N column vectors
def flatten(arr):
    output = arr.reshape((arr.shape[0], arr.shape[1] * arr.shape[2]))
    output = np.einsum('ij->ji', output)
    return output


# simple neural network architecture with 784 input neurons, 1 hidden layer of 20 neurons, and 1 output neuron
class AntonMNIST:
    # parameter initialization
    def __init__(self, input_units=784, def_hidden_units=20):
        rng1 = np.random.default_rng(5)
        self.w1 = rng1.standard_normal((def_hidden_units, input_units)) * np.sqrt(2 / input_units).astype(np.float32)
        self.b1 = np.zeros((def_hidden_units, 1)).astype(np.float32)
        self.w2 = rng1.standard_normal((10, def_hidden_units)) * np.sqrt(2 / def_hidden_units).astype(np.float32)
        self.b2 = np.zeros((10, 1)).astype(np.float32)

        # adding velocities for momentum
        self.vdw1 = np.zeros_like(self.w1)
        self.vdb1 = np.zeros_like(self.b1)
        self.vdw2 = np.zeros_like(self.w2)
        self.vdb2 = np.zeros_like(self.b2)

        # store activations for backpropogation and gradients for update
        self.activations = {}
        self.gradients = {}

    # rectified linear unit activation function
    @staticmethod
    def relu(inputs):
        input_copy = np.copy(inputs)
        input_copy[(input_copy < 0)] = 0
        return input_copy

    # gradients of activation functions
    @staticmethod
    def relu_derivative(inputs):
        return (inputs > 0).astype(float)

    # softmax function
    @staticmethod
    def softmax(z):
        z_shifted = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    # forward pass 2x takes input matrix, multiplies weights and add biases, then applies activation function
    def forward(self, inputs):
        z1 = np.dot(self.w1, inputs) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = self.softmax(z2)

        # saves intermediate values for backpropogation
        self.activations["a0"] = inputs
        self.activations["z1"] = z1
        self.activations["a1"] = a1
        self.activations["z2"] = z2
        self.activations["a2"] = a2

        return a2

    # cross-entropy loss function
    @staticmethod
    def calculate_loss(probs, labels1):
        m = labels1.shape[0]
        probs = np.clip(probs, 1e-12, 1.0)
        correct_class_probs = probs[labels1, np.arange(m)]
        loss = -np.mean(np.log(correct_class_probs))
        return loss

    # backpropogation = how much each parameter contributed to the loss via gradients
    def calculate_gradients(self, labels2):
        m = labels2.shape[0]

        y_one_hot = one_hot(labels2, num_classes=10)

        dz2 = self.activations["a2"] - y_one_hot
        dw2 = 1 / m * np.dot(dz2, self.activations["a1"].T)
        db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.multiply(self.w2.T @ dz2, self.relu_derivative(self.activations["z1"]))
        dw1 = 1 / m * np.dot(dz1, self.activations["a0"].T)
        db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)

        self.gradients["dz2"] = dz2
        self.gradients["dz1"] = dz1
        self.gradients["dw1"] = dw1
        self.gradients["db1"] = db1
        self.gradients["dw2"] = dw2
        self.gradients["db2"] = db2

    # update parameters based on the gradients from backpropogation and the learning rate (lr)
    def update_parameters(self, learning_rate, beta=0.9):
        self.vdw1 = beta * self.vdw1 + (1 - beta) * np.square(self.gradients["dw1"])
        self.w1 += - learning_rate * self.gradients["dw1"] / (np.sqrt(self.vdw1) + 1e-8)
        self.vdb1 = beta * self.vdb1 + (1 - beta) * np.square(self.gradients["db1"])
        self.b1 += - learning_rate * self.gradients["db1"] / (np.sqrt(self.vdb1) + 1e-8)
        self.vdw2 = beta * self.vdw2 + (1 - beta) * np.square(self.gradients["dw2"])
        self.w2 += - learning_rate * self.gradients["dw2"] / (np.sqrt(self.vdw2) + 1e-8)
        self.vdb2 = beta * self.vdb2 + (1 - beta) * np.square(self.gradients["db2"])
        self.b2 += - learning_rate * self.gradients["db2"] / (np.sqrt(self.vdb2) + 1e-8)


# create batches of size batch_size from x and y
def get_batches(batch_size1, x, y):
    batches1 = []
    m = y.shape[0]

    i = 0
    while i < m:
        start = i
        end = i + batch_size1

        if end >= m:
            batches1.append((x[:, start:], y[start:]))
        else:
            batches1.append((x[:, start:end], y[start:end]))

        i += batch_size1

    return batches1


# calculate the accuracy of logits (predictions) vs. labels=y_hat (actual)
def accuracy(probs, y_hat):
    predicted_labels = np.argmax(probs, axis=0)
    return np.mean(predicted_labels == y_hat)

def one_hot(labels2, num_classes=10):
    output = np.zeros((num_classes, labels2.shape[0]))
    output[labels2, np.arange(labels2.shape[0])] = 1
    return output


def train_model(
        lr_,
        hu_,
        bs_,
        ne
):
    training_model = AntonMNIST(def_hidden_units=hu_)

    for _ in range(ne):
        indices_ = np.arange(x_train_scaled_shuffled.shape[1])
        rng_ = np.random.default_rng(5)
        rng_.shuffle(indices_)

        x_epoch = x_train_scaled_shuffled[:, indices_]
        y_epoch = y_train_shuffled[indices_]

        train_batches = get_batches(
            bs_,
            x_epoch,
            y_epoch
        )

        for train_features, train_labels in train_batches:
            training_model.forward(train_features)

            training_model.calculate_gradients(train_labels)
            training_model.update_parameters(lr_)

    train_dev_logits = training_model.forward(x_dev_scaled)

    return {
        "dev_loss": training_model.calculate_loss(train_dev_logits, y_dev),
        "dev_acc": accuracy(train_dev_logits, y_dev)
    }




if __name__ == "__main__":

    start = time.time()

    # load in train, dev, and test datasets
    x_train, y_train = read_images("data/fashion_mnist/train/")
    x_dev, y_dev = read_images("data/fashion_mnist/dev/")
    x_test, y_test = read_images("data/fashion_mnist/test/")

    print("Loading: ", time.time() - start)

    # compute the dataset statistics
    dataset_mean, dataset_std = x_train.mean(), x_train.std()

    x_train_scaled = normalize(x_train, dataset_mean, dataset_std).astype(np.float32)
    x_dev_scaled = normalize(x_dev, dataset_mean, dataset_std).astype(np.float32)
    x_test_scaled = normalize(x_test, dataset_mean, dataset_std).astype(np.float32)

    x_train_scaled = flatten(x_train_scaled)
    x_dev_scaled = flatten(x_dev_scaled)
    x_test_scaled = flatten(x_test_scaled)

    # shuffle training data
    indices = np.arange(x_train_scaled.shape[1])
    rng = np.random.default_rng(5)
    rng.shuffle(indices)

    x_train_scaled_shuffled = x_train_scaled[:, indices]
    y_train_shuffled = y_train[indices]

    # hyperparameters (external parameters set by engineer)
    learning_rates = [0.001]
    hidden_units = [200]
    batch_sizes = [512]

    results = []

    for lr in learning_rates:
        for hu in hidden_units:
            for bs in batch_sizes:

                start = time.time()

                metrics = train_model(
                    lr_=lr,
                    hu_=hu,
                    bs_=bs,
                    ne=10
                )

                results.append({
                    "lr": lr,
                    "hidden_units": hu,
                    "batch_size": bs,
                    "dev_acc": metrics["dev_acc"],
                    "dev_loss": metrics["dev_loss"],
                    "loop_time": time.time() - start
                })

                print(results[-1])