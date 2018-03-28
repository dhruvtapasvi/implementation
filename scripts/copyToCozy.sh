#!/usr/bin/env bash

#scp -o "GSSAPIAuthentication=yes" -r src ddt21@cozy.cl.cam.ac.uk:~/implementation
#scp -o "GSSAPIAuthentication=yes" -r rough ddt21@cozy.cl.cam.ac.uk:~/implementation
#scp -o "GSSAPIAuthentication=yes" -r scripts ddt21@cozy.cl.cam.ac.uk:~/implementation
#scp -o "GSSAPIAuthentication=yes" -r config ddt21@cozy.cl.cam.ac.uk:~/implementation

#scp -o "GSSAPIAuthentication=yes" cacheWeights/norb_pca_500_2048_1_10_0_fitted_variance.weights.h5 ddt21@cozy.cl.cam.ac.uk:~/implementation/src/

scp -o "GSSAPIAuthentication=yes" -r pcaDownsample ddt21@cozy.cl.cam.ac.uk:~/implementation/
