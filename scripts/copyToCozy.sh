#!/usr/bin/env bash

scp -o "GSSAPIAuthentication=yes" -r src ddt21@cozy.cl.cam.ac.uk:~/implementation
scp -o "GSSAPIAuthentication=yes" -r scripts ddt21@cozy.cl.cam.ac.uk:~/implementation