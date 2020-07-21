# DnCNN - TensorFlow 2   
A TensorFlow 2 implementation of the paper [Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://arxiv.org/pdf/1608.03981.pdf)

## Try in Colab..
If you don't have a GPU, you can use the notebook `dncnn.ipynb` in Google Colab for both training and testing :)

## Dataset
In this implementation the patches are not precomputed. Instead, at each epoch, a random patch is extracted from each image, and we need to train for more epochs.

At test time, the center 180x180 crop of each image is used.

The dataset is the [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500). The train and test folder are merged to create the 400 images training set, while the 100 images (which contain the 68 images from the BSD68 dataset) in the valid folder are used as test set. The model is also tested on the 12 popular images for the image denoising benchmark (which can be found in `data/set12`).
## Model Architecture
![graph](/img/model.png)


## Results
![compare](/img/img_7.png)

- BSD100 (containing BSD68) Average Result 
 

|  Noise Level | DnCNN-TensorFlow2 |
|:------------:|:-----------------:|
|      25      |     **27.37**     |

- Set12 Average Result


| Noise Level | DnCNN-TensorFlow2 |
|:-----------:|:-----------------:|
| 25          |    **27.41**      |



## Requirements
```
pip install -r requirements.txt
```
The code was written in Python 3.7
## Train
```
$ python train.py
(note: You can add command line arguments according to the source code, for example
    $ python main.py --batch_size 64 )
```
You can monitor loss and psnr for both training and test set with tensorboard:
```
tensorboard --logdir logs
```

## Test
By default, the script will use the weights in `weights/vgg` from the pretrained network, and the network is tested on the set12 dataset.
```
$ python test.py
(note: Also here you can add command line arguments, such as --save_plots to save the plots for the results)
```

## Quantization
By default, the script will use the model in `saved_models/vgg` from the pretrained network.

Among the other command line arguments, you can use `--psnr` to compute the psnr for the quantized network (by default on set12).

If you add `--no_q` the model without quantization will also be saved and tested for comparison.
```
$ python quantization.py
```

## Quantization Results on set12

|  Noise Level | DnCNN-TensorFlow2-quantized |
|:------------:|:---------------------------:|
|      25      |          **27.35**          |

Network size reduced from 2.3MB to 610KB.



