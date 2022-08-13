# Before you read
This is a student collaboration project which requirement is finding a topic and reproducing a related research paper.
The research theory and related code are downloaded from https://github.com/tamarott/SinGAN
The code executing instructions are copied from authors repo. We add a shell script to train all the input images with one commmand.
# CS523 Project: Recreation of SinGAN

## Background of single image training technique for GAN and related research
Texture Expansion
(Zhou, Y., Zhu, Z., Bai, X., Lischinski, D., Cohen-Or, D., & Huang, H. (2018). Non-Stationary Texture Synthesis by Adversarial Expansion. ArXiv:1805.04487 [Cs]. https://arxiv.org/abs/1805.04487)

Training GAN with single image already come into play for a while. 
In 2018, there’s a research about texture expansion. It relies on single texture image, and achieve high quality picture expansion by generating more texture to fill up the image. 

InGAN-Internal GAN
(Shocher, A., Bagon, S., Isola, P., & Irani, M. (n.d.). InGAN: Capturing and Retargeting the “DNA” of a Natural Image. Retrieved August 10, 2022, from https://openaccess.thecvf.com/content_ICCV_2019/papers/Shocher_InGAN_Capturing_and_Retargeting_the_DNA_of_a_Natural_Image_ICCV_2019_paper.pdf)
By 2019, there’s another research which based on single natural image and achieve retargeting. Retargeting at here can be understand by two aspects, the first parts is expanding, the second part is shrinking. Both action should not tilt, scale or change the major target in the image. A good example you can see here, the input is a picture with three birds in it, and on the right hand side, when we we change the width of the image, instead of squiz the picture, the output from InGAN just put three birds moves closer to each other. 

SinGAN
(Shaham, T., Dekel, T., Research, G., & Michaeli, T. (n.d.). SinGAN: Learning a Generative Model from a Single Natural Image. Retrieved August 10, 2022, from https://openaccess.thecvf.com/content_ICCV_2019/papers/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.pdf)
At 2019 we have a better solution, SinGAN. Why it’s better? First SinGAN is unconditional veras InGAN is conditional. Secondly application wise, SinGAN can also achieve highper resolution, harmonization. We can also create animation from single image. 
We show that the internal statistics of patches within a single natural image typically carry enough information for learning a powerful generative model. Once trained, SinGAN can produce diverse high quality image samples (of arbitrary dimensions), which semantically resemble the training image, yet contain new object configurations and structures. 

## Idea of SinGAN Reiteration
The architecture of SinGAN is a pyramid of fully convolutional GANS, each capture the distribution of patches at a different scale of an image. When training a model, the image is downsampled at each scale. The generation of an image starts at the coarsest scale and passes through generators up to the finest scale, with noise injected at every scale. The receptive field of each generator and discriminator is the same, so smaller size of structure is captured as the generation process goes up. In many of our training examples, we try to generate image starting at different scales after training the original target images. At the coarsest scale, generated images have large variation with the original image; at finer scale, SinGAN only modifies details, like textures of the sky or stipes on the zebras. 

## Our Work

In this project, we trained around 40 images in total. The images are images provided by the author, BSD100 dataset, and from our own photos (all in output folder). For many images, we try to further present the same images generated at different scales for comparison (all in output - RandomSamples folder). We also show several examples for each application (all in folders below output). Moreover, we try to adjust four hyperparameters to SinGAN model: pyramid scale factor, reconstruction loss weight, number of layers, and number of epochs to train per scale. The first two hyperparameters investigate the effect of changes on quality of generated images. The last two hyperparameters decide the training speed of the model and see if training faster still produces images of the same quality. We try these changes of hyperparameters on two images: Starry Night and zebra. Starry Night is art work, so probably any modification is allowed; in comparison, zebra is an image of real life so it has to look real, otherwise the generated image is a failure. We want to compare if art work and real-life image require different hyperparameters so that we can have 2 models to train them differently in future.

To evaluate the effect of different hyperparameters and make conclusion, we have 3 difficulties:

1.	Evaluation is quite subjective. Someone in our group has different opinion on realness and fakeness of two sets of images generated by different hyperparameters. Also, is absolute different from the original input or similar from the original input image better if they look both real? The answer depends on individuals.
2.	As said before, if we generate images at coarse scale, the generated image has great variability in global structures, while generate at finer scale only changes finer details. Therefore, if we change one parameter, like reconstruction loss weight, and compare, we may find that the image generated at smaller weight but at lower scale may have the similar distribution than the image generated at larger weight but higher scale. We cannot distinguish which reconstruction loss weight is better at this case.
3.	Each training model would gives us 50 random samples. To almost all the parameters, there are good examples or bad examples. If we only compare the best example generated by different parameters, they are likely to be awesome. Therefore, it is hard to say which parameter is better.

Therefore, in our following analysis, we would emphasize the difference of parameters. Even if we evaluate which parameter is better, it will still be our subjective opinion.
 
### Changed parameter1 - Scale Factor
The model was trained with scale factors of 0.25, 0.5, 0.75 (default), and 0.85 with the following command.
```
python main_train.py --scale_factor <scale factor> --input_name <input file name> 
``` 
Two images, zebra.png and starry_night.png, were tested. In both cases, the models trained with a scale factor of 0.25 captured finer details in the original images, but not their global structures.

As scale factor increases, the models were able to better capture the global structures of the input images.


### Changed parameter2

### Changed parameter3 - Number of Layers
The model was trained with 4, 5 (default), and 6 layers with the following command. 
```
python main_train.py --num_layer <number of layers> --input_name <input file name> 
``` 
Two images, zebra.png and starry_night.png, were tested. In both cases, the models trained with 4 layers captured the distribution of the input images poorly.

The models with 5 layers were sufficient to capture the distribution of the original images. 

The models with 6 layers produced random samples that were nearly identical to the original images with only minor differences.


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
python main_train.py --input_name <input_file_name>
```

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

To transfer a paint into a realistic image (See example in Fig. 11 in (https://arxiv.org/pdf/1905.01164.pdf)), please first train SinGAN model on the desired image (as described above), then save your paint under "Input/Paint", and run the command

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

## Our Addition and findings
### Train all the image
You can trian all the image under Input/Image directory by running `Run.sh`

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

###
