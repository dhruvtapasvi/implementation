#!/usr/bin/env bash

cd res
mkdir norb
cd norb

curl https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz -o norb_train_image.gz
curl https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz -o norb_train_category.gz
curl https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz -o norb_train_info.gz
curl https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz -o norb_test_image.gz
curl https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz -o norb_test_category.gz
curl https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz -o norb_test_info.gz

gunzip norb_train_image.gz
gunzip norb_train_category.gz
gunzip norb_train_info.gz
gunzip norb_test_image.gz
gunzip norb_test_category.gz
gunzip norb_test_info.gz

cd ../..
