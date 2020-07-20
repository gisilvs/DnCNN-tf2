import argparse
import os

import matplotlib.pyplot as plt
import tensorflow as tf

from dncnn import DnCNN
from dncnnrn import DnCNNRN

parser = argparse.ArgumentParser(description='DnCNN tf2 test')
parser.add_argument('--model', default='DnCNN', choices=['DnCNN', 'DnCNNRN'], type=str, help='choose a type of model')
parser.add_argument('--data_dir', default='data/set12', type=str, help='path of test data')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--depth', default=17, type=int, help='depth of the model')
parser.add_argument('--test_size', default=180, type=int, help='size for test images')
parser.add_argument('--batch_size', default=12, type=int, help='batch size')
parser.add_argument('--format', default='png', choices=['jpg', 'png'], type=str, help='image format')
parser.add_argument('--weights_path', default='weights/vgg', type=str, help='path for loading model weights')
parser.add_argument('--save_plots', action='store_true', help='save plots in plots_dir')
parser.add_argument('--plots_dir', default='plots', type=str, help='path for saving plots')

args = parser.parse_args()

AUTOTUNE = tf.data.experimental.AUTOTUNE  # for dataset configuration

# Network parameters
MODEL = args.model
DEPTH = args.depth

# Data preparation variables
NOISE_STD = args.sigma
TEST_DIM = args.test_size
FORMAT = args.format
BATCH_SIZE = args.batch_size

# Train and test set directories
TEST_DIR = args.data_dir + '/*.' + FORMAT

# Paths for saving weights and plots
WEIGHTS_PATH = args.weights_path
SAVE_PLOTS = args.save_plots
PLOTS_DIR = args.plots_dir

if MODEL == 'DnCNN':
    model = DnCNN(depth=DEPTH)
elif MODEL == 'DnCNNRN':
    model = DnCNNRN(depth=DEPTH)

model.load_weights(WEIGHTS_PATH)


def gaussian_noise_layer(dim):
    '''generate noise mask of given dimension'''
    std = NOISE_STD
    noise = tf.random.normal(shape=[dim, dim, 1], mean=0.0, stddev=std, dtype=tf.float32) / 255.0
    return noise


def augment(image):
    image = tf.io.read_file(image)
    if FORMAT == 'jpg':
        image = tf.image.decode_jpeg(image, channels=1)
    elif FORMAT == 'png':
        image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize_with_crop_or_pad(image, TEST_DIM, TEST_DIM)
    image = tf.image.convert_image_dtype(image, tf.float32)
    noise = gaussian_noise_layer(TEST_DIM)
    noisy_image = tf.clip_by_value(image + noise, 0, 1)
    return noisy_image, image


def configure_ds(ds):
    ds = ds.cache()
    ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


@tf.function
def test(images, targets):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    predictions = tf.clip_by_value(predictions, 0, 1)
    t_metric = tf.image.psnr(predictions, targets, max_val=1.0)

    test_metric(t_metric)


test_metric = tf.keras.metrics.Mean(name='test_metric')
test_metric.reset_states()
test_ds_list = tf.data.Dataset.list_files(TEST_DIR)
test_ds = configure_ds(test_ds_list)

for test_images, test_targets in test_ds:
    test(test_images, test_targets)

print(f'Avreage PSNR: {test_metric.result().numpy()}')

if SAVE_PLOTS:
    # Plot the results
    # from left to right:
    # noisy image, prediction, true image
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    image_batch, target_batch = next(iter(test_ds))
    predictions = model(image_batch, training=False)
    predictions = tf.clip_by_value(predictions, 0, 1)
    test_ds_list = list(test_ds_list)

    for i in range(len(test_ds_list)):
        f, axarr = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        plt.sca(axarr[0])
        plt.imshow(image_batch[i].numpy().squeeze(), cmap='gray')
        plt.axis("off")
        plt.title('Noisy')
        plt.sca(axarr[1])
        plt.imshow(predictions[i].numpy().squeeze(), cmap='gray')
        plt.axis("off")
        plt.title('Prediction')
        plt.sca(axarr[2])
        plt.imshow(target_batch[i].numpy().squeeze(), cmap='gray')
        plt.axis("off")
        plt.title('Original')
        plt.savefig(f'{PLOTS_DIR}/img_{i + 1}.png')
