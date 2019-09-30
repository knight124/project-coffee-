import gzip
import math

import numpy as np
import matplotlib.pyplot as plt


def loadData(address, numimages, sizeImages):
    f = gzip.open(address, 'r')
    images = []
    f.read(16)
    buf = f.read(numimages * sizeImages * sizeimages)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

    data = data.reshape(numimages, sizeImages, sizeImages, 1)

    for x in data:
        images.append(np.asarray(x).squeeze())

    return images


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


if __name__ == "__main__":
    addresstraining = 'train-images-idx3-ubyte.gz'
    addresslables = 'train-labels-idx1-ubyte.gz'
    numimages = 60000
    sizeimages = 28

    training_data = np.asarray(loadData(addresstraining, numimages, sizeimages))

    hl1 = convolve(3, 8, 1)
    pl1 = pool(5)
    hl2 = convolve(3, 16, 0)
    pl2 = pool(3)
    hl3 = convolve(5, 8, 0)

    out = hl1.forward(training_data[0])

    out = pl1.pooling(out)
    # print(out.shape)
    out = hl2.forward(out)
    print(out.shape)
    out = pl2.pooling(out)

    out = hl3.forward(out)

    pl3 = pool(out.shape[1])
    out = pl3.pooling(out)
    print(out.shape)

