#!/bin/bash

echo "Starting E part"
python e.py
echo "Starting A part"
python a.py
echo "Starting test.py with base model resnet"
python test.py --model resnet --file_extend base
echo "Starting test.py with base model vit"
python test.py --model vit --file_extend base
echo "Starting test.py with base model resnet and 100 layers"
python test.py --model resnet --file_extend 100_l --layers 100
echo "Starting test.py with base model vit and 100 layers"
python test.py --model vit --file_extend 100_l --layers 100
