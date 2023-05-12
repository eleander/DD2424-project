#!/bin/bash

python e.py
python a.py
python test.py --model resnet --file_extend base
python test.py --model vit --file_extend base
python test.py --model resnet --file_extend 100_l --layers 100
python test.py --model vit --file_extend 100_l --layers 100
