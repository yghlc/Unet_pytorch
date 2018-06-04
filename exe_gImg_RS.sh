#!/bin/bash


export CUDA_VISIBLE_DEVICES=1

para_file=para.ini
para_py=/home/hlc/codes/PycharmProjects/DeeplabforRS/parameters.py
eo_dir=$(python2 ${para_py} -p ${para_file} codes_dir)
expr=${PWD}
testid=$(basename $expr)

data_dir=/home/hlc/Data/Qinghai-Tibet/beiluhe/beiluhe_google_img

rm train_loss.txt

#  run our pre-trained model provided by user ,  commit:
#python main.py unsData --resume checkpoint_BN.tar --niter 0 --useBN 
#--cuda


# run train 

#python main.py unsData --worker 1 --batchSize 8 --niter 25 --lr 0.0002 --cuda --useBN --output_name checkpoint_1.tar

#train 
#python main.py ${data_dir} para.ini train_list.txt  --worker 8 --batchSize 8 --niter 200 --lr 0.0002 --cuda --useBN --output_name checkpoint_gImg_v4.tar

#exit

### run test
rm -r inf_result
mkdir inf_result
python test.py ${data_dir} para.ini inf_list.txt  --worker 1 --batchSize 1 --cuda --useBN --resume checkpoint_gImg_v4.tar

### post processing
cd inf_result
    gdal_merge.py -init 0 -n 0 -a_nodata 0 -o ${testid}_out.tif *_pred.tif
cd ..

rm -r post_pro_val_result
mkdir post_pro_val_result
cd post_pro_val_result
mv ../inf_result/${testid}_out.tif .
# convert to shapefile
gdal_polygonize.py -8 ${testid}_out.tif -b 1 -f "ESRI Shapefile" ${testid}.shp
cd ..


