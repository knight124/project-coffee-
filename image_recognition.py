import gzip
import math

import numpy as np
import matplotlib.pyplot as plt


def loadData(address, numimages, sizeImages):
    f = gzip.open(address, 'r')
    l = gzip.open('train-labels-idx1-ubyte.gz', 'r')
    images = []
    lables = []
    l.read(8)
    f.read(16)
    buf = f.read(numimages * sizeImages * sizeimages)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

    data = data.reshape(numimages, sizeImages, sizeImages, 1)
    for i in range(0, numimages):
        buf = l.read(1)
        lables.append(np.frombuffer(buf, dtype=np.uint8).astype(np.int64))
    for x in data:
        images.append(np.asarray(x).squeeze())

    return images, lables


# class NN:
#     def __init__(self,num_inputs):
#         self. numberofinputs = num_inputs
#         self.num_hidden = 20
#         self.num_hidden1 = 10
#         self.num_out = 10
#
#         self.weights1 = np.random.rand (self.numberofinputs,self.num_hidden)
#         self.weights2 = np.random.rand(self.num_hidden, self.num_hidden1)
#         self.weights3 = np.random.rand(self.num_hidden1,self.num_out)
#
#     def sig (self,x):
#         return 1/(1+np.exp(-x))
#
#     def divsig (self, x):
#         return x * (1-x)
#
#     def forward (self, x):
#         self.intermidiat  = np.dot(x,self.weights1)
#         self.intermidiat2 = self.sig(self.intermidiat)
#         self.intermidiat3 = np.dot(self.intermidiat2, self.weights2)
#         self.intermidiat4 = self.sig(self.intermidiat3)
#         self.intermidiat5 = np.dot(self.intermidiat4, self.weights3)
#         out = self.sig(self.intermidiat5)
#
#     def backward (self, x, y, o):
#         self.output_error = y-o
#         self.output_delta = self.o_error*self.divsig(o)
#
#         self.inermidiate4_error = self.output_delta.dot(self.weights3)
#         self.intermidiat4_delta = self.inermidiate4_error*self.divsig(self.intermidiat4)
#
#         self.intermidiat2.error = self.intermidiat4_delta(self.weights2.T)
#         self.intermidiat2_delta = self.inermidiate2_error*self.divsig(self.intermidiat4)
#
#         self.weights1 += x.T.dot(self.intermidiat2_delta)
#         self.weights2 += self.intermidiat4.T.dot(self.intermidiat4_delta)
#         self.weights3 += self.intermidiat2.T.dot(self.intermidiat2_delta)


class Softmax:
    # A standard fully-connected layer with softmax activation.

    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.lastinshape = input.shape
        input = input.flatten()
        self.last_in = input
        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases

        self.last_total = totals
        exp = np.exp(totals)

        return exp / np.sum(exp, axis=0)

    def backprop(self, gradloss, lr):
        for i, gradient in enumerate(gradloss):
            if gradient == 0:
                continue
            t_exp = np.exp(self.last_total)
            Sum = np.sum(t_exp)

            gradout = -t_exp[i] * t_exp / (Sum ** 2)
            gradout[i] = t_exp[i] * (Sum - t_exp[i]) / (Sum ** 2)

            gradw = self.last_in
            gradb = 1
            gradin = self.weights
            gradl = gradient * gradout
            gradw = gradw[np.newaxis].T @ gradl[np.newaxis]
            gradb = gradl * gradb
            gradin = gradin @ gradl

            self.weights -= lr * gradw
            self.biases -= lr * gradb

            return gradin.reshape(self.lastinshape)


class convolve:

    def __init__(self, size, num_filters, isinput):
        self.num_filters = num_filters
        self.size = size
        self.isinput = isinput
        self.filters = np.random.rand(num_filters, size, size) / (size * size)

    def iteration_reg(self, image):

        width, height = image.shape

        for i in range(height - (self.size - 1)):
            for j in range(width - (self.size - 1)):
                reg = image[i:(i + self.size), j:(j + self.size)]

                yield reg, i, j

    def forward(self, input):
        if self.isinput:
            width, height = input.shape
            num_input = 1
        else:
            num_input, width, height = input.shape
            self.filters = np.random.rand(self.num_filters * num_input, self.size, self.size) / (self.size * self.size)

        out = np.zeros((self.num_filters * num_input, height - (self.size - 1), width - (self.size - 1)))
        if self.isinput:

            for reg, i, j in self.iteration_reg(input):
                out[:, i, j] = np.sum(reg * self.filters, axis=(1, 2))
        else:
            for x in range(num_input):
                for reg, i, j in self.iteration_reg(input[x, :, :]):
                    out[x * self.num_filters:x * self.num_filters + self.num_filters, i, j] = np.sum(
                        reg * self.filters[x * self.num_filters:x * self.num_filters + self.num_filters, :, :],
                        axis=(1, 2))

        return out


class pool:

    def __init__(self, size):

        self.size = size

    def iterat(self, image):
        reg = 0
        # print(image.shape)
        _, width, height = image.shape
        for i in range(height - (self.size - 1)):
            for j in range(width - (self.size - 1)):
                reg = np.amax(image[:, i:(i + self.size), j:(j + self.size)], axis=(1, 2))

                yield reg, i, j

    def pooling(self, input):
        num_filters, width, height = input.shape
        out = np.zeros((num_filters, height - (self.size - 1), width - (self.size - 1)))

        for reg, i, j in self.iterat(input):
            out[:, i, j] = reg

        return out


def forward(image, label):
    out = hl1.forward(image/255 -0.5 )
    # image123 = np.asarray(out[0]).squeeze()
    # plt.imshow(image123)
    # plt.show()
    out = pl1.pooling(out)
    # image123 = np.asarray(out[0]).squeeze()
    # plt.imshow(image123)
    # plt.show()
    # print(out.shape)
    # out = hl2.forward(out)
    # out = pl2.pooling(out)
    # out = hl3.forward(out)
    # out = pl3.pooling(out)

    out = softmax.forward(out)
   # print (out)

    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc


def train(im, lable, lr=0.005):
    out, l, acc = forward(im, lable)

    gradient = np.zeros(10)
    gradient[lable] = -1 / out[lable]
    gradient = softmax.backprop(gradient, lr)
    return l, acc


if __name__ == "__main__":
    addresstraining = 'train-images-idx3-ubyte.gz'
    addresslables = 'train-labels-idx1-ubyte.gz'
    numimages = 60000
    sizeimages = 28
    loss = 0
    num_correct = 0

    training_data, lable_data = np.asarray(loadData(addresstraining, numimages, sizeimages))

    hl1 = convolve(3, 8, 1)  # 28x28x1 ->26x26x8
    pl1 = pool(14)  # 26x26x8 -> 22x22x8
    hl2 = convolve(3, 8, 0)  # 22x22x8 -> 20X20X64
    pl2 = pool(3)  # 20x20x128 -> 18x18x 128
    hl3 = convolve(5, 8, 0)  # 18x18x128 -> 14x14x1024
    pl3 = pool(11)  # 14x14x1024 ->4x4x1024
    softmax = Softmax(13*13*8, 10)

for i in range(numimages):
    # print(training_data[i], lable_data[i])
    l, acc = train(training_data[i], lable_data[i])
    loss += l
    num_correct += acc

    if i % 100 == 99:
        print('[Step%d] past 100 step: Average Loss %.3f | Accuracy %d%%' % (i + 1, loss / 100, num_correct))
        loss = 0
        num_correct = 0
