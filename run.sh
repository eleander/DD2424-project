#!/bin/bash

echo "--- Starting E part ---"
python e.py
echo "--- Starting A part ---"
python a.py
echo "--- Starting Extra part ---"
python extra.py --model resnet --file_extend base
python extra.py --model vit --file_extend base
python extra.py --model vit --n_labels 150 --file_extend base150
python extra.py --model vit --n_labels 100 --file_extend base100
python extra.py --model vit --n_labels 50 --file_extend base50

python extra.py --dataset oxfordiiitpet --model resnet --file_extend noaug 