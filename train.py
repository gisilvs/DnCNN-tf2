import argparse
import datetime
import random

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.losses import MeanSquaredError

from dncnn import DnCNN
from dncnnrn import DnCNNRN

parser = argparse.ArgumentParser(description='DnCNN tf2')
parser.add_argument('--model', default='DnCNN', choices=['DnCNN', 'DnCNNRN'], type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--train_data', default='data/train', type=str, help='path of train data')
parser.add_argument('--test_data', default='data/test', type=str, help='path of test data')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epochs', default=500, type=int, help='number of train epochs')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--wd', default=0.0001, type=float, help='weight decay')
parser.add_argument('--depth', default=17, type=int, help='depth of the model')
parser.add_argument('--train_patch', default=40, type=int, help='size for training patches')
parser.add_argument('--test_size', default=180, type=int, help='size for test images')
parser.add_argument('--format', default='jpg', choices=['jpg', 'png'], type=str, help='image format')
parser.add_argument('--weights_path', default='weights/vgg', type=str, help='path for saving model weights')
parser.add_argument('--model_path', default='saved_models/vgg', type=str, help='path for saving whole model')
parser.add_argument('--exp_name', default='vgg', type=str, help='name for experiment logs')

args = parser.parse_args()

AUTOTUNE = tf.data.experimental.AUTOTUNE  # for dataset configuration

# Training variables
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.lr
WEIGHT_DECAY = args.wd

# Network parameters
MODEL = args.model
DEPTH = args.depth

# Data preparation variables
NOISE_STD = args.sigma
SCALES = [1, 0.9, 0.8, 0.7]  # used for data augmentation
TRAIN_PATCH_DIM = args.train_patch
TEST_DIM = args.test_size
FORMAT = args.format

# Train and test set directories
TEST_DIR = args.test_data + '/*.' + FORMAT
TRAIN_DIR = args.train_data + '/*' + FORMAT

# Paths for saving weights and model
WEIGHTS_PATH = args.weights_path
MODEL_PATH = args.model_path

# Tensorboard logs name
EXPERIMENT_NAME = args.exp_name


def gaussian_noise_layer(dim):
    '''generate noise mask of given dimension'''
    std = NOISE_STD  # random.randint(0, 55) for blind denoising
    noise = tf.random.normal(shape=[dim, dim, 1], mean=0.0, stddev=std, dtype=tf.float32) / 255.0
    return noise


def augment(image):
    '''prepare and augment input image, and generate noise mask'''
    image = tf.io.read_file(image)

    if FORMAT == 'jpg':
        image = tf.image.decode_jpeg(image, channels=1)
    elif FORMAT == 'png':
        image = tf.image.decode_png(image, channels=1)

    # augmentation 1:
    # rescale input to obtain crops at different level of detail
    h, w = float(tf.shape(image)[0]), float(tf.shape(image)[1])
    s = random.choice(SCALES)
    image = tf.image.resize(image, [int(h * s), int(w * s)]) / 250

    # crop random patch
    image = tf.image.random_crop(image, size=[TRAIN_PATCH_DIM, TRAIN_PATCH_DIM, 1])

    # augmentation 2: random flip
    image = tf.image.random_flip_left_right(image)

    # augmentation 3: random rotation (0, 90, 180 or 270 degrees)
    for i in range(np.random.randint(4)):
        image = tf.image.rot90(image)

    # generate noise mask
    noise = gaussian_noise_layer(TRAIN_PATCH_DIM)

    # sum image and noise, clip values between 0 and 1
    noisy_image = tf.clip_by_value(image + noise, 0, 1)

    return noisy_image, image


def configure_for_train(ds):
    '''configure loading and batching for training set'''
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=100)
    ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def augment_test(image):
    '''load test image'''

    # No augmentation and different size from training
    image = tf.io.read_file(image)

    if FORMAT == 'jpg':
        image = tf.image.decode_jpeg(image, channels=1)
    elif FORMAT == 'png':
        image = tf.image.decode_png(image, channels=1)

    image = tf.image.resize_with_crop_or_pad(image, 180, 180)
    image = tf.image.convert_image_dtype(image, tf.float32)
    noise = gaussian_noise_layer(180)
    noisy_image = tf.clip_by_value(image + noise, 0, 1)

    return noisy_image, image


def configure_for_test(ds):
    '''configure loading and batching for training set'''

    # No random shuffle
    ds = ds.cache()
    ds = ds.map(augment_test, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


@tf.function
def train_step(images, targets):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(targets, predictions)
        predictions = tf.clip_by_value(predictions, 0, 1)
        metric = tf.image.psnr(predictions, targets, max_val=1.0)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_metric(metric)


@tf.function
def test_step(images, targets):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(targets, predictions)
    predictions = tf.clip_by_value(predictions, 0, 1)
    t_metric = tf.image.psnr(predictions, targets, max_val=1.0)

    test_loss(t_loss)
    test_metric(t_metric)


# Load train and test set
train_ds = tf.data.Dataset.list_files(TRAIN_DIR)
train_ds = configure_for_train(train_ds)
test_ds = tf.data.Dataset.list_files(TEST_DIR)
test_ds = configure_for_test(test_ds)

# define model, loss function and optimizer
if MODEL == 'DnCNN':
    model = DnCNN(depth=DEPTH)
elif MODEL == 'DnCNNRN':
    model = DnCNNRN(depth=DEPTH)

loss_object = MeanSquaredError()
optimizer = tfa.optimizers.AdamW(weight_decay=WEIGHT_DECAY, learning_rate=LEARNING_RATE)

# these objects keep track of losses and metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_metric = tf.keras.metrics.Mean(name='train_metric')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_metric = tf.keras.metrics.Mean(name='test_metric')

# Set tensorflow dir for the experiment
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time + '_' + EXPERIMENT_NAME
summary_writer = tf.summary.create_file_writer(log_dir)

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_metric.reset_states()
    test_loss.reset_states()
    test_metric.reset_states()

    # training loop
    for images, targets in train_ds:
        train_step(images, targets)

    # test loop
    for test_images, test_targets in test_ds:
        test_step(test_images, test_targets)

    # log losses and metrics
    with summary_writer.as_default():
        tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
        tf.summary.scalar('train_psnr', train_metric.result(), step=epoch)
        tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
        tf.summary.scalar('test_psnr', test_metric.result(), step=epoch)

# need weights to load the model for inference
model.save_weights(WEIGHTS_PATH)

# need whole model for post-training quantization
model.save(MODEL_PATH)
