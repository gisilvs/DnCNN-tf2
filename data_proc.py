import random

import matplotlib.pyplot as plt
import tensorflow as tf
from dncnn import DnCNN
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam



def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return img  # tf.image.resize(img, [img_height, img_width])


def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


def gaussian_noise_layer():
    std = random.randint(0, 55)
    noise = tf.random.normal(shape=[40, 40, 1], mean=0.0, stddev=std / 255.0, dtype=tf.float32)
    return noise


def augment(image):
    image = tf.io.read_file(image)
    image = decode_img(image)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.random_crop(image, size=[40, 40, 1])
    image = tf.image.convert_image_dtype(image, tf.float32)
    noise = gaussian_noise_layer()

    return image + noise, noise, image

@tf.function
def train_step(images, targets):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)


batch_size = 64
img_height = 40
img_width = 40
AUTOTUNE = tf.data.experimental.AUTOTUNE

data_dir = 'data/BSDS500/data/images/train/*.jpg'

train_ds = tf.data.Dataset.list_files(data_dir)


# train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=100)
    ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_ds = configure_for_performance(train_ds)

image_batch, target_batch, true_batch = next(iter(train_ds))

'''plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy().squeeze(), cmap='gray')
    plt.axis("off")'''

model = DnCNN()

loss_object = MeanSquaredError()

optimizer = Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')

EPOCHS = 5

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()

    for images, targets, true in train_ds:
        train_step(images, targets)
        print('ciao')

    template = 'Epoch {}, Loss: {}'
    print(template.format(epoch + 1,
                          train_loss.result()))
