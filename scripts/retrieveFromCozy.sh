#!/usr/bin/env bash

scp -o "GSSAPIAuthentication=yes" ddt21@cozy.cl.cam.ac.uk:~/implementation/cacheWeights/*.h5 cacheWeights/
scp -o "GSSAPIAuthentication=yes" ddt21@cozy.cl.cam.ac.uk:~/implementation/modelTrainingHistory/*.p modelTrainingHistory/
