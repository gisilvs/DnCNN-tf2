from tensorflow.keras import Model
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU


class DnCNNRN(Model):
    def __init__(self, depth=8):
        super(DnCNNRN, self).__init__()

        # Initial conv + relu (same as in DnCNN)
        self.conv = Conv2D(64, 3, padding='same', kernel_initializer=he_uniform())
        self.relu = ReLU()

        # Use 8 ResNet-inspired blocks (16 layers)
        self.rn_layers = [BasicBlock() for i in range(depth)]

        # Final conv
        self.conv_final = Conv2D(1, 3, padding='same', kernel_initializer=he_uniform())

    def call(self, x):
        out = self.conv(x)
        out = self.relu(out)
        for layer in self.rn_layers:
            out = layer(out)
        return x - self.conv_final(out)


class BasicBlock(Model):
    def __init__(self):
        # One ResNet block is:
        # conv1 - bn1 - relu
        # conv2 - bn2
        # residual connection
        # relu

        super(BasicBlock, self).__init__()

        self.conv1 = Conv2D(64, 3, padding='same', kernel_initializer=he_uniform())
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        self.conv2 = Conv2D(64, 3, padding='same', kernel_initializer=he_uniform())
        self.bn2 = BatchNormalization()

    def call(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity

        out = self.relu(out)

        return out
