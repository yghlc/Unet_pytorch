#!/usr/bin/env bash


org_dir="/home/hlc/Data/Qinghai-Tibet/beiluhe/beiluhe_google_img/BLH_deeplab_google_6"

para_file=para.ini
para_py=/home/hlc/codes/PycharmProjects/DeeplabforRS/parameters.py

eo_dir=$(python2 ${para_py} -p ${para_file} codes_dir)

#### preparing training images
#crop image to fix size
folder="split_images"

mkdir $folder
cd $folder

for img in ${org_dir}/${folder}/*.tif
do
    filename=$(basename $img)
    #gdal_translate -srcwin 0 0 580 420 $img $filename
done
cd ..

###preparing training labels
folder="split_labels"
mkdir $folder
cd $folder

for img in ${org_dir}/${folder}/*.tif
do
    filename=$(basename $img)
    #gdal_translate -srcwin 0 0 580 420 -a_nodata 0 $img $filename
done
cd ..

### preparing inference images
# inference the same image of input images for training
RSimg=$(python2 ${para_py} -p ${para_file} input_image_path)
folder="inf_split_images"

rm -r $folder
mkdir $folder

patch_w=$(python2 ${para_py} -p ${para_file} inf_patch_width)     # the expected width of patch
patch_h=$(python2 ${para_py} -p ${para_file} inf_patch_height)     # the expected height of patch
overlay=$(python2 ${para_py} -p ${para_file} inf_pixel_overlay)     # the overlay of patch in pixel

###pre-process inf images
${eo_dir}/split_image.py -W ${patch_w} -H ${patch_h} -e ${overlay} -o ${PWD}/inf_split_images ${RSimg}

# make sure all the images have the same size
cd $folder
for img in *.tif
do
    filename=$(basename $img)
    gdal_translate -srcwin 0 0 560 400 -a_nodata 0 $img tmp.tif
    mv tmp.tif $img
done
cd ..
