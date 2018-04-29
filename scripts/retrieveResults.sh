#!/usr/bin/env bash

scp -o "GSSAPIAuthentication=yes" ddt21@cozy.cl.cam.ac.uk:~/implementation/out/* out/
scp -o "GSSAPIAuthentication=yes" ddt21@cozy.cl.cam.ac.uk:~/implementation/results/* results/
