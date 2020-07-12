from tensorflow.keras import Model
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU


class DnCNN(Model):
    def __init__(self, depth=20, grayscale=True):
        super(DnCNN, self).__init__()
        if grayscale:
            self.conv1 = Conv2D(64, 1, padding='same', activation='relu', kernel_initializer=he_uniform())
        else:
            self.conv1 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer=he_uniform())

        self.conv_bn_relu = [ConvBNReLU() for i in range(depth - 2)]

        if grayscale:
            self.conv_final = Conv2D(1, 64, padding='same', kernel_initializer=he_uniform())
        else:
            self.conv_final = Conv2D(3, 64, padding='same', kernel_initializer=he_uniform())

    def call(self, x):
        x = self.conv1(x)
        #for cbr in self.conv_bn_relu:
            #x = cbr(x)
        return self.conv_final(x)


class ConvBNReLU(Model):
    def __init__(self):
        super(ConvBNReLU, self).__init__()
        self.conv = Conv2D(64, 64, padding='same', kernel_initializer=he_uniform())
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)
