
# CS523 Project: Recreation of SinGAN

Team members: Chun Cheng, Kaiyan Xu, Hui Zheng
## Idea of SinGAN Reiteration
The architecture of SinGAN is a pyramid of fully convolutional GANS, each capture the distribution of patches at a different scale of an image. When training a model, the image is downsampled at each scale. The generation of an image starts at the coarsest scale and passes through generators up to the finest scale, with noise injected at every scale. The receptive field of each generator and discriminator is the same, so smaller size of structure is captured as the generation process goes up. In many of our training examples, we try to generate image starting at different scales after training the original target images. At the coarsest scale, generated images have large variation with the original image; at finer scale, SinGAN only modifies details, like textures of the sky or stipes on the zebras. 


## Code Instruction

### Install dependencies

To run the code on scc, load the following modules:

cuda/10.1

python3/3.6.9

pytorch/1.3


###  Train

All the images used for training are located under "Input/Images."
To train the model, use the following command and replace <input_file_name> with the name of a file under "Input/Images."
```
python main_train.py --input_name <input_file_name>
```

###  Random samples

The following command generates 50 random samples of <training_image_file_name> starting from <generation_start_scale_number>. 

(This command can only be used after training <training_image_file_name> using the command above. <generation_start_scale_number> can be any integer between 0 to N, where N is the last scale trained during training.)
```
python random_samples.py --input_name <training_image_file_name> --mode random_samples --gen_start_scale <generation_start_scale_number>
```

###  Random samples of arbitrary sizes

The following command performs in the same way as the command above, except that it allows specification of output size.
```
python random_samples.py --input_name <training_image_file_name> --mode random_samples_arbitrary_sizes --scale_h <horizontal scaling factor> --scale_v <vertical scaling factor>
```

###  Animation from a single image

The following generates an animation (.gif) from <input_file_name>.

(<input_file_name> does not need to be trained before calling this command. <input_file_name> should be located under "Input/Images.")
```
python animation.py --input_name <input_file_name> 
```

###  Harmonization

The following harmonizes an object <naively_pasted_reference_image_file_name> into the background image <training_image_file_name>.

(Before calling this command, we must train the model using <training_image_file_name> first. <naively_pasted_reference_image_file_name> should be located under "Input/Harmonization." <scale to inject> can be any integer between 0 to N, where N is the last scale trained during training.)
```
python harmonization.py --input_name <training_image_file_name> --ref_name <naively_pasted_reference_image_file_name> --harmonization_start_scale <scale to inject>

```

###  Editing

The following modifies an edited image <edited_image_file_name>.

(Before calling this command, we must train the model using the unedited image <training_image_file_name> first. <edited_image_file_name> should be located under "Input/Editing." <scale to inject> can be any integer between 0 to N, where N is the last scale trained during training.)
```
python editing.py --input_name <training_image_file_name> --ref_name <edited_image_file_name> --editing_start_scale <scale to inject>

``` 

###  Paint to Image

The following turns a paint <paint_image_file_name> into a realistic image based on <training_image_file_name>.

(Before calling this command, we must train the model using a realistic image <training_image_file_name> first. <paint_image_file_name> should be located under "Input/Paint." <scale to inject> can be any integer between 0 to N, where N is the last scale trained during training.)
```
python paint2image.py --input_name <training_image_file_name> --ref_name <paint_image_file_name> --paint_start_scale <scale to inject>

```

### Super Resolution

The following turns a low resolution image <LR_image_file_name> into high resolution.

(<LR_image_file_name> does not need to be trained before calling this command. <LR_image_file_name> should be located under "Input/Images.")
```
python SR.py --input_name <LR_image_file_name>
```
### Output
All the outputs from the commands above are located in "Output" folder.


## Our Work

In this project, we trained xxx images in total, from the original input, bsd 100 dataset, and from our own photos. For many images, we try to further present the same images generated at different scales for comparison. We also show several examples for each application. Moreover, we try to adjust four hyperparameters to SinGAN model: pyramid scale factor, reconstruction loss weight, number of layers, and number of epochs to train per scale. The first two hyperparameters investigate change in quality of generated images. The last two hyperparameters decide the training speed of the model and see if using lower value still produces the same qualified images. We try these changes in hyperparameters on two images: Starry Night and zebra. Starry night is art work and may allow more variation or “modification” so its strictness is lower; zebra is image of real life so it has to look real, otherwise the generated image is a failure. We want to compare if art work and real-life image require different hyperparameters so that in future, we can have 2 models to train them differently (all output in #output# folder).

To evaluate the effect of different hyperparameters and make conclusion, we have 3 difficulties:

1.	Evaluation is quite subjective. Someone in our group has different opinion on realness and fakeness of two sets of images generated by different hyperparameters. Also, is absolute different from the original input or similar from the original input image better if they look both real? The answer depends on individuals.
2.	As said before, if we generate images at coarse scale, the generated image has great variability in global structures, while generate at finer scale only changes finer details. Therefore, if we change one parameter, like reconstruction loss weight, and compare, we may find that the image generated at smaller weight but at lower scale may have the similar distribution than the image generated at larger weight but higher scale. We cannot distinguish which reconstruction loss weight is better at this case.
3.	Each training model would gives us 50 random samples. To almost all the parameters, there are good examples or bad examples. If we only compare the best example generated by different parameters, they are likely to be awesome. Therefore, it is hard to say which parameter is better.

Therefore, in our following analysis, we would emphasize the difference of parameters. Even if we evaluate which parameter is better, it will still be our subjective opinion.

### Changed parameter1

### Changed parameter2

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
  
## Slides
  
## Reference
