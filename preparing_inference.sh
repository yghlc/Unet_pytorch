#!/usr/bin/env bash



para_file=para.ini
para_py=/home/hlc/codes/PycharmProjects/DeeplabforRS/parameters.py

eo_dir=$(python2 ${para_py} -p ${para_file} codes_dir)
root=$(python2 ${para_py} -p ${para_file} working_root)

# current folder (without path)
test_dir=${PWD##*/}

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
min_size=692340
for img in *.tif
do
    size=$( stat -c%s $img )
    if [ "$size" -lt  "$min_size" ]
    then
        echo $img
        gdal_translate -srcwin 0 0 480 480 -a_nodata 0 $img temp.tif
        mv temp.tif $img
    fi
done
