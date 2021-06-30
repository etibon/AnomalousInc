#!/bin/bash

sudo docker run -p 8889:8888 -v `pwd`/jupyter/src:/src etienne/lp-jupyter
