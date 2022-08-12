#!/bin/sh
filepath=Input/Images/*.png
for file in $filepath
do 
	filename=$(basename $file)
	 python main_train.py --input_name $filename
done 
