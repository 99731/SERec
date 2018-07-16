#!/bin/bash
result_path=/home/wmh/c_sexpo
home_path=$result_path/mlast
train_user=$home_path/train_user
train_item=$home_path/train_item
vali_data=$home_path/vali_data
test_data=$home_path/test_data
social_data=$home_path/social_data
cd /home/wmh/c_sexpo
for version in 1       
do
./sexpo --version $version --directory $result_path --user $train_user --item $train_item --social $social_data --vali_data $vali_data --test_data $test_data --a 1 --b 1 --s 250 --lambda_theta 0.01 --lambda_beta 0.01 --lambda_mu 0.1 --max_iter 20 --n_components 30 --metric_k 50
done 
