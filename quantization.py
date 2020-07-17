import argparse
from pathlib import Path

import tensorflow as tf

parser = argparse.ArgumentParser(description='DnCNN tf2 test')
parser.add_argument('--model_dir', default='saved_models/vgg', type=str, help='path containing saved model')
parser.add_argument('--data_dir', default='data/set12', type=str, help='path of test data')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--test_size', default=180, type=int, help='size for test images')
parser.add_argument('--format', default='png', choices=['jpg', 'png'], type=str, help='image format')
parser.add_argument('--tflite_dir', default='tflite_models/', type=str, help='path for saving optimized models')
parser.add_argument('--psnr', action='store_true', help='compute psnr for test data')
parser.add_argument('--no_q', action='store_true', help='use also non-qunatized model for comparison')

args = parser.parse_args()

AUTOTUNE = tf.data.experimental.AUTOTUNE  # for dataset configuration

# Data preparation variables
NOISE_STD = args.sigma
TEST_DIM = args.test_size
FORMAT = args.format

# Train and test set directories
TEST_DIR = args.data_dir + '/*.' + FORMAT

# Paths for saving weights and plots
MODEL_PATH = args.model_dir
TFLITE_DIR = args.tflite_dir
NON_Q = args.no_q
PSNR = args.psnr
MODEL_NAME = MODEL_PATH.split('/')[-1]


def evaluate_model(interp, noises):
    '''A helper function to evaluate the TF Lite model using "test" dataset.'''
    input_index = interp.get_input_details()[0]["index"]
    output_index = interp.get_output_details()[0]["index"]

    # Run predictions on every image in the "test" dataset.
    psnr = 0.0
    for i, image in enumerate(test_ds):
        noisy_image = tf.clip_by_value(image + noises[i], 0, 1)
        interp.set_tensor(input_index, noisy_image)

        # Run inference.
        interp.invoke()

        output = tf.clip_by_value(interp.get_tensor(output_index), 0, 1)
        psnr += tf.image.psnr(output, image, max_val=1.0)

    return psnr / (i + 1)


def gaussian_noise_layer(dim):
    '''generate noise mask of given dimension'''
    std = NOISE_STD
    noise = tf.random.normal(shape=[dim, dim, 1], mean=0.0, stddev=std, dtype=tf.float32) / 255.0
    return noise


def augment_test(image):
    image = tf.io.read_file(image)
    if FORMAT == 'jpg':
        image = tf.image.decode_jpeg(image, channels=1)
    elif FORMAT == 'png':
        image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize_with_crop_or_pad(image, TEST_DIM, TEST_DIM)
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image


def configure_for_test(ds):
    ds = ds.cache()
    ds = ds.map(augment_test, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(1)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


# load saved model as TFLiteConverter
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_PATH)

# create output dir
tflite_models_dir = Path(TFLITE_DIR)
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# code for converting model without quantization
# same as for quantized model
if NON_Q:
    tflite_model = converter.convert()
    tflite_model_file = Path(tflite_models_dir, MODEL_NAME + ".tflite")
    tflite_model_file.write_bytes(tflite_model)

# convert model with quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# save model
tflite_model_quant_file = Path(tflite_models_dir, MODEL_NAME + "_quant.tflite")
tflite_model_quant_file.write_bytes(tflite_quant_model)

if PSNR:

    # compute psnr on test set

    tf.random.set_seed(123)
    test_dir = TEST_DIR

    # configure test set
    test_ds = tf.data.Dataset.list_files(test_dir)

    # create one noise mask for each image
    # used in both quantized and not quantized model for fair comparison
    len_ds = len(list(test_ds))
    test_ds = configure_for_test(test_ds)
    noises = [gaussian_noise_layer(TEST_DIM) for i in range(len_ds)]

    if NON_Q:
        # same as for quantized model
        interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
        input_details = interpreter.get_input_details()
        interpreter.resize_tensor_input(input_details[0]['index'], (1, TEST_DIM, TEST_DIM, 1))
        interpreter.allocate_tensors()
        psnr = evaluate_model(interpreter, noises)
        print(f'Avreage PSNR standard model: {psnr.numpy()[0]}')

    # create interpreter for quantized model
    interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))

    # adjust size of accepted input
    input_details_quant = interpreter_quant.get_input_details()
    interpreter_quant.resize_tensor_input(input_details_quant[0]['index'], (1, TEST_DIM, TEST_DIM, 1))

    # predict test set with quantized model
    interpreter_quant.allocate_tensors()
    psnr = evaluate_model(interpreter_quant, noises)
    print(f'Avreage PSNR quantized model: {psnr.numpy()[0]}')
