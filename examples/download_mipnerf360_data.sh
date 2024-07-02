#!/bin/bash

wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip -O 360_v2.zip
unzip 360_v2.zip -d data/360_v2
rm 360_v2.zip