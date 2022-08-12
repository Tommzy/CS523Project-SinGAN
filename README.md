
# CS523 Project: Recreation of SinGAN
## Idea of SinGAN Reiteration
The architecture of SinGAN is a pyramid of fully convolutional GANS, each capture the distribution of patches at a different scale of an image. When training a model, the image is downsampled at each scale. The generation of an image starts at the coarsest scale and passes through generators up to the finest scale, with noise injected at every scale. The receptive field of each generator and discriminator is the same, so smaller size of structure is captured as the generation process goes up. In many of our training examples, we try to generate image starting at different scales after training the original target images. At the coarsest scale, generated images have large variation with the original image; at finer scale, SinGAN only modifies details, like textures of the sky or stipes on the zebras. 

## Our Work

In this project, we trained xxx images in total, from the original input, bsd 100 dataset, and from our own photos. For many images, we try to further present the same images generated at different scales for comparison. We also show several examples for each application. Moreover, we try to adjust four hyperparameters to SinGAN model: pyramid scale factor, reconstruction loss weight, number of layers, and number of epochs to train per scale. The first two hyperparameters investigate change in quality of generated images. The last two hyperparameters decide the training speed of the model and see if using lower value still produces the same qualified images. We try these changes in hyperparameters on two images: Starry Night and zebra. Starry night is art work and may allow more variation or “modification” so its strictness is lower; zebra is image of real life so it has to look real, otherwise the generated image is a failure. We want to compare if art work and real-life image require different hyperparameters so that in future, we can have 2 models to train them differently (all output in #output# folder).

To evaluate the effect of different hyperparameters and make conclusion, we have 3 difficulties:

1.	Evaluation is quite subjective. Someone in our group has different opinion on realness and fakeness of two sets of images generated by different hyperparameters. Also, is absolute different from the original input or similar from the original input image better if they look both real? The answer depends on individuals.
2.	As said before, if we generate images at coarse scale, the generated image has great variability in global structures, while generate at finer scale only changes finer details. Therefore, if we change one parameter, like reconstruction loss weight, and compare, we may find that the image generated at smaller weight but at lower scale may have the similar distribution than the image generated at larger weight but higher scale. We cannot distinguish which reconstruction loss weight is better at this case.
3.	Each training model would gives us 50 random samples. To almost all the parameters, there are good examples or bad examples. If we only compare the best example generated by different parameters, they are likely to be awesome. Therefore, it is hard to say which parameter is better.

Therefore, in our following analysis, we would emphasize the difference of parameters. Even if we evaluate which parameter is better, it will still be our subjective opinion.

### Changed parameter1

### Changed parameter2

### Changed parameter3

### Changed parameter4

## Code Instruction
## Code

### Install dependencies

To run the code on scc, load the following modules:

cuda/10.1

python3/3.6.9

pytorch/1.3


###  Train
To train SinGAN model on your own image, put the desired training image under Input/Images, and run

```
python main_train.py --input_name <input_file_


###  Random samples
To generate random samples from any starting generation scale, please first train SinGAN model on the desired image (as described above), then run 

```
python random_samples.py --input_name <training_image_file_name> --mode random_samples --gen_start_scale <generation start scale number>
```

pay attention: for using the full model, specify the generation start scale to be 0, to start the generation from the second scale, specify it to be 1, and so on. 

###  Random samples of arbitrary sizes
To generate random samples of arbitrary sizes, please first train SinGAN model on the desired image (as described above), then run 

```
python random_samples.py --input_name <training_image_file_name> --mode random_samples_arbitrary_sizes --scale_h <horizontal scaling factor> --scale_v <vertical scaling factor>
```

###  Animation from a single image

To generate short animation from a single image, run

```
python animation.py --input_name <input_file_name> 
```

This will automatically start a new training phase with noise padding mode.

###  Harmonization

To harmonize a pasted object into an image (See example in Fig. 13 in [our paper](https://arxiv.org/pdf/1905.01164.pdf)), please first train SinGAN model on the desired background image (as described above), then save the naively pasted reference image and it's binary mask under "Input/Harmonization" (see saved images for an example). Run the command

```
python harmonization.py --input_name <training_image_file_name> --ref_name <naively_pasted_reference_image_file_name> --harmonization_start_scale <scale to inject>

```

Please note that different injection scale will produce different harmonization effects. The coarsest injection scale equals 1. 

###  Editing

To edit an image, (See example in Fig. 12 in [our paper](https://arxiv.org/pdf/1905.01164.pdf)), please first train SinGAN model on the desired non-edited image (as described above), then save the naive edit as a reference image under "Input/Editing" with a corresponding binary map (see saved images for an example). Run the command

```
python editing.py --input_name <training_image_file_name> --ref_name <edited_image_file_name> --editing_start_scale <scale to inject>

```
both the masked and unmasked output will be saved.
Here as well, different injection scale will produce different editing effects. The coarsest injection scale equals 1. 

###  Paint to Image

To transfer a paint into a realistic image (See example in Fig. 11 in [our paper](https://arxiv.org/pdf/1905.01164.pdf)), please first train SinGAN model on the desired image (as described above), then save your paint under "Input/Paint", and run the command

```
python paint2image.py --input_name <training_image_file_name> --ref_name <paint_image_file_name> --paint_start_scale <scale to inject>

```
Here as well, different injection scale will produce different editing effects. The coarsest injection scale equals 1. 

Advanced option: Specify quantization_flag to be True, to re-train *only* the injection level of the model, to get a on a color-quantized version of upsampled generated images from the previous scale. For some images, this might lead to more realistic results.

### Super Resolution
To super resolve an image, please run:
```
python SR.py --input_name <LR_image_file_name>
```
This will automatically train a SinGAN model correspond to 4x upsampling factor (if not exist already).
For different SR factors, please specify it using the parameter `--sr_factor` when calling the function.
SinGAN's results on the BSD100 dataset can be download from the 'Downloads' folder.

## Additional Data and Functions

### Single Image Fréchet Inception Distance (SIFID score)
To calculate the SIFID between real images and their corresponding fake samples, please run:
```
python SIFID/sifid_score.py --path2real <real images path> --path2fake <fake images path> 
```  
Make sure that each of the fake images file name is identical to its corresponding real image file name. Images should be saved in `.jpg` format.

## Our Addition

### Number of Layers
The model was trained with 4, 5 (default), and 6 layers with the following command. 
```
python main_train.py --num_layer <number of layers> --input_name <input file name> 
``` 
Two images, zebra.png and starry_night.png, were tested. In both cases, the models trained with 4 layers captured the distribution of the input images poorly.

The models with 5 layers were sufficient to capture the distribution of the original images. 

The models with 6 layers produced random samples that were nearly identical to the original images with only minor differences.

### Scale Factor
The model was trained with scale factors of 0.25, 0.5, 0.75 (default), and 0.85 with the following command.
```
python main_train.py --scale_factor <scale factor> --input_name <input file name> 
``` 
Two images, zebra.png and starry_night.png, were tested. In both cases, the models trained with a scale factor of 0.25 captured finer details in the original images, but not their global structures.

As scale factor increases, the models were able to better capture the global structures of the input images.
