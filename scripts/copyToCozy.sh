#!/usr/bin/env bash

#scp -o "GSSAPIAuthentication=yes" -r src ddt21@cozy.cl.cam.ac.uk:~/implementation
#scp -o "GSSAPIAuthentication=yes" -r rough ddt21@cozy.cl.cam.ac.uk:~/implementation
#scp -o "GSSAPIAuthentication=yes" -r scripts ddt21@cozy.cl.cam.ac.uk:~/implementation
#scp -o "GSSAPIAuthentication=yes" -r config ddt21@cozy.cl.cam.ac.uk:~/implementation

scp -o "GSSAPIAuthentication=yes" pca/norb_pca_500.p ddt21@cozy.cl.cam.ac.uk:~/implementation/pca/
