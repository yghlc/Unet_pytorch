#!/usr/bin/env bash



para_file=para.ini
para_py=/home/hlc/codes/PycharmProjects/DeeplabforRS/parameters.py

eo_dir=$(python2 ${para_py} -p ${para_file} codes_dir)
root=$(python2 ${para_py} -p ${para_file} working_root)

# current folder (without path)
test_dir=${PWD##*/}

rm -r ${root}/${test_dir}/split_images
rm -r ${root}/${test_dir}/split_labels
mkdir  ${root}/${test_dir}/split_images ${root}/${test_dir}/split_labels

#### preparing training images

input_train_image=$(python2 ${para_py} -p ${para_file} input_train_image)
# input groud truth (raster data with pixel value)
input_GT=$(python2 ${para_py} -p ${para_file} input_ground_truth_image)

### split the training image to many small patch (480*480)
patch_w=$(python2 ${para_py} -p ${para_file} train_patch_width)
patch_h=$(python2 ${para_py} -p ${para_file} train_patch_height)
overlay=$(python2 ${para_py} -p ${para_file} train_pixel_overlay)     # the overlay of patch in pixel

trainImg_dir=$(python2 ${para_py} -p ${para_file} input_train_dir)
labelImg_dir=$(python2 ${para_py} -p ${para_file} input_label_dir)

for img in ${trainImg_dir}/*.tif
do
${eo_dir}/split_image.py -W ${patch_w} -H ${patch_h}  -e ${overlay} -o  ${root}/${test_dir}/split_images $img
done
for img in ${labelImg_dir}/*.tif
do
${eo_dir}/split_image.py -W ${patch_w} -H ${patch_h}  -e ${overlay} -o ${root}/${test_dir}/split_labels $img
done


## remove ( enlarge ) files which is not 480*480

# make sure all the images have the same size
out_w=$(python2 ${para_py} -p ${para_file} out_patch_width)
out_h=$(python2 ${para_py} -p ${para_file} out_patch_height)

cd split_images
for img in *.tif
do
    size=$(gdalinfo ${img} | grep "Size is" )
    width=$(echo $size | cut -d' ' -f 3 )
    width=${width::-1}
    height=$(echo $size | cut -d' ' -f 4 )
    echo "*****width,height*****:" $width , $height
    if [ "$width" -lt  $out_w -o $height -lt $out_h ]
    then
        echo $img
        gdal_translate -srcwin 0 0 $out_w $out_h -a_nodata 0 $img temp.tif
        mv temp.tif $img
    fi
done


cd ../split_labels

for img in *.tif
do
    size=$(gdalinfo ${img} | grep "Size is" )
    width=$(echo $size | cut -d' ' -f 3 )
    width=${width::-1}
    height=$(echo $size | cut -d' ' -f 4 )
    echo "*****width,height*****:" $width , $height
    if [ "$width" -lt  $out_w -o $height -lt $out_h ]
    then
        echo $img
        gdal_translate -srcwin 0 0 $out_w $out_h -a_nodata 0 $img temp.tif
        mv temp.tif $img
    fi
done

cd ..




