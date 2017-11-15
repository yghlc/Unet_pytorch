#!/usr/bin/env bash



para_file=para.ini
para_py=/home/hlc/codes/PycharmProjects/DeeplabforRS/parameters.py

eo_dir=$(python2 ${para_py} -p ${para_file} codes_dir)
root=$(python2 ${para_py} -p ${para_file} working_root)

# current folder (without path)
test_dir=${PWD##*/}

#rm -r ${root}/${test_dir}/split_images
#rm -r ${root}/${test_dir}/split_labels
#mkdir  ${root}/${test_dir}/split_images ${root}/${test_dir}/split_labels

#### preparing training images

input_train_image=$(python2 ${para_py} -p ${para_file} input_train_image)
# input groud truth (raster data with pixel value)
input_GT=$(python2 ${para_py} -p ${para_file} input_ground_truth_image)

### split the training image to many small patch (480*480)
patch_w=$(python2 ${para_py} -p ${para_file} train_patch_width)
patch_h=$(python2 ${para_py} -p ${para_file} train_patch_height)
overlay=$(python2 ${para_py} -p ${para_file} train_pixel_overlay)     # the overlay of patch in pixel

#${eo_dir}/split_image.py -W ${patch_w} -H ${patch_h}  -e ${overlay} -o  ${root}/${test_dir}/split_images $input_train_image
#${eo_dir}/split_image.py -W ${patch_w} -H ${patch_h}  -e ${overlay} -o ${root}/${test_dir}/split_labels $input_GT

# remove ( enlarge ) files which is not 480*480

cd split_images

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
min_size=231004

cd ../split_labels

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

cd ..




