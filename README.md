# CS523 Project: Recreation of SinGAN


## Code

### Install dependencies

To run the code on scc, load the following modules:

cuda/10.1

python3/3.6.9

pytorch/1.3


###  Train

```
python main_train.py --input_name <input_file_name>
```


###  Random samples

```
python random_samples.py --input_name <training_image_file_name> --mode random_samples --gen_start_scale <generation start scale number>
```

###  Random samples of arbitrary sizes

```
python random_samples.py --input_name <training_image_file_name> --mode random_samples_arbitrary_sizes --scale_h <horizontal scaling factor> --scale_v <vertical scaling factor>
```

###  Animation from a single image

```
python animation.py --input_name <input_file_name> 
```

###  Harmonization

```
python harmonization.py --input_name <training_image_file_name> --ref_name <naively_pasted_reference_image_file_name> --harmonization_start_scale <scale to inject>

```

###  Editing

```
python editing.py --input_name <training_image_file_name> --ref_name <edited_image_file_name> --editing_start_scale <scale to inject>

``` 

###  Paint to Image

```
python paint2image.py --input_name <training_image_file_name> --ref_name <paint_image_file_name> --paint_start_scale <scale to inject>

```

### Super Resolution

```
python SR.py --input_name <LR_image_file_name>
```

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
