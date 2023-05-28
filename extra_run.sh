#!/bin/bash
# Extra experiments for the paper

echo "Testing different un-freezing blocks"
python extra.py --model resnet --dataset oxfordiiitpet --file_extend block_1 --unfreeze_blocks 1
python extra.py --model resnet --dataset oxfordiiitpet --file_extend block_2 --unfreeze_blocks 2
python extra.py --model resnet --dataset oxfordiiitpet --file_extend block_3 --unfreeze_blocks 3
python extra.py --model resnet --dataset oxfordiiitpet --file_extend block_4 --unfreeze_blocks 4

echo "Testing different classifier layers"
python extra.py --model resnet --dataset oxfordiiitpet --file_extend layers_100 --layers 100
python extra.py --model vit --dataset oxfordiiitpet --file_extend layers_100 --layers 100
python extra.py --model resnet --dataset oxfordiiitpet --file_extend layers_200_100 --layers 200 --layers 100
python extra.py --model vit --dataset oxfordiiitpet --file_extend layers_200_100 --layers 200 --layers 100
python extra.py --model resnet --dataset oxfordiiitpet --file_extend layers_300_200_100 --layers 300 --layers 200 --layers 100
python extra.py --model vit --dataset oxfordiiitpet --file_extend layers_300_200_100 --layers 300 --layers 200 --layers 100

echo "Testing different un-freezing blocks"
python extra.py --model resnet --dataset oxfordiiitpet --file_extend lr_0.0001 --lr 1e-4
python extra.py --model vit --dataset oxfordiiitpet --file_extend lr_0.0001 --lr 1e-4
python extra.py --model resnet --dataset oxfordiiitpet --file_extend lr_0.01 --lr 1e-2
python extra.py --model vit --dataset oxfordiiitpet --file_extend lr_0.01 --lr 1e-2
python extra.py --model resnet --dataset oxfordiiitpet --file_extend lr_0.00001 --lr 1e-5
python extra.py --model vit --dataset oxfordiiitpet --file_extend lr_0.00001 --lr 1e-5
