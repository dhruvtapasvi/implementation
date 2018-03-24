#!/usr/bin/env bash

# Actual
#scp -o "GSSAPIAuthentication=yes" ddt21@cozy.cl.cam.ac.uk:~/implementation/src/*.h5 cacheWeights/
#scp -o "GSSAPIAuthentication=yes" ddt21@cozy.cl.cam.ac.uk:~/implementation/src/*.p modelTrainingHistory/

# Rough
scp -o "GSSAPIAuthentication=yes" ddt21@cozy.cl.cam.ac.uk:~/implementation/src/*.png out/
scp -o "GSSAPIAuthentication=yes" ddt21@cozy.cl.cam.ac.uk:~/implementation/src/*.p pca/
scp -o "GSSAPIAuthentication=yes" ddt21@cozy.cl.cam.ac.uk:~/implementation/src/*.h5 cacheWeights/

