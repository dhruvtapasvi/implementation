#!/usr/bin/env bash

scp -o "GSSAPIAuthentication=yes" ddt21@cozy.cl.cam.ac.uk:~/implementation/src/*.h5 cacheWeights/
scp -o "GSSAPIAuthentication=yes" ddt21@cozy.cl.cam.ac.uk:~/implementation/src/*.p modelTrainingHistory/

