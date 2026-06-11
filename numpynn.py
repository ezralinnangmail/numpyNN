import numpy as np

np.random.seed(5)

import os
from PIL import Image


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
    def __init__(self, input_units=784, hidden_units=20):
        rng1 = np.random.default_rng(5)
        self.w1 = rng1.standard_normal((hidden_units, input_units)) * np.sqrt(2 / input_units)
        self.b1 = np.zeros((hidden_units, 1))
        self.w2 = rng1.standard_normal((1, hidden_units)) * np.sqrt(2 / hidden_units)
        self.b2 = np.zeros((1, 1))

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
        relu_grad = np.copy(inputs)
        relu_grad[(relu_grad >= 0)] = 1
        relu_grad[(relu_grad < 0)] = 0
        assert (relu_grad == (inputs >= 0).astype(float)).all()
        return relu_grad

    # forward pass 2x takes input matrix, multiplies weights and add biases, then applies activation function
    def forward(self, inputs):
        z1 = np.dot(self.w1, inputs) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = 1 / (1 + np.exp(-z2))

        # saves intermediate values for backpropogation
        self.activations["a0"] = inputs
        self.activations["z1"] = z1
        self.activations["a1"] = a1
        self.activations["z2"] = z2
        self.activations["a2"] = a2

        return a2

    # binary cross-entropy loss function
    @staticmethod
    def calculate_loss(logits, labels1):
        m = labels1.shape[0]
        logits = np.clip(logits, 1e-7, 1 - 1e-7)
        losses = -(np.multiply(labels1, np.log(logits)) + np.multiply((1 - labels1), np.log(1 - logits)))
        total_loss = np.sum(losses / m)
        return total_loss

    # backpropogation = how much each parameter contributed to the loss via gradients
    def calculate_gradients(self, labels2):
        m = labels2.shape[0]
        dz2 = self.activations["a2"] - labels2
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
    def update_parameters(self, lr):
        self.w1 += - lr * self.gradients["dw1"]
        self.b1 += - lr * self.gradients["db1"]
        self.w2 += - lr * self.gradients["dw2"]
        self.b2 += - lr * self.gradients["db2"]


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
def accuracy(logits, y_hat):
    predicted_labels = (logits > 0.5).astype(float)
    num_examples = y_hat.shape[0]
    correct_predictions = (predicted_labels == y_hat).sum().item()
    return correct_predictions / num_examples


if __name__ == "__main__":
    # load in train, dev, and test datasets
    x_train, y_train = read_images("data/binary_mnist/train/")
    x_dev, y_dev = read_images("data/binary_mnist/dev/")
    x_test, y_test = read_images("data/binary_mnist/test/")

    # compute the dataset statistics
    dataset_mean, dataset_std = x_train.mean(), x_train.std()

    x_train_scaled = normalize(x_train, dataset_mean, dataset_std)
    x_dev_scaled = normalize(x_dev, dataset_mean, dataset_std)
    x_test_scaled = normalize(x_test, dataset_mean, dataset_std)

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
    learning_rate = 0.0001
    batch_size = 32

    batches = get_batches(batch_size, x_train_scaled_shuffled, y_train_shuffled)
    model = AntonMNIST()

    # plotting
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    train_loss_history = []
    dev_loss_history = []


    num_epochs = 10
    for epoch in range(num_epochs):
        for batch_idx, (features, labels) in enumerate(batches):
            if np.any(np.isnan(features)):
                print(f"NaN detected in features at batch {batch_idx}")
                break
            if np.any(np.isnan(labels)):
                print(f"NaN detected in labels at batch {batch_idx}")
                break

            # forward pass
            output_logits = model.forward(features)
            train_loss = model.calculate_loss(output_logits, labels)

            # backwards pass
            model.calculate_gradients(labels)
            model.update_parameters(learning_rate)

            dev_logits = model.forward(x_dev_scaled)
            dev_loss = model.calculate_loss(dev_logits, y_dev)
            dev_acc = accuracy(dev_logits, y_dev)

            # logging
            if batch_idx % (len(batches) // 5) == 0:
                train_loss_history.append(train_loss)
                dev_loss_history.append(dev_loss)
'''
                print(f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
                      f" | Batch {batch_idx:03d}/{len(batches):03d}"
                      f" | Train Loss: {loss:.3f}")

                print(f"Dev Loss: {dev_loss:.4f} | Dev Accuracy: {dev_acc:.4f}")
'''

plt.plot(train_loss_history, marker='o')
plt.plot(dev_loss_history, marker='x')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.show()

# formal testing
test_output_logits = model.forward(x_test_scaled)
loss_test = model.calculate_loss(test_output_logits, y_test)
acc_test = accuracy(test_output_logits, y_test)
print(f"Loss on testing dataset: {loss_test}")
print(f"Accuracy on testing dataset: {acc_test}")
