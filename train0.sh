#!/bin/bash
#SBATCH --job-name=sa_isic2020         # create a short name for your job
#SBATCH --constraint=gpu80
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=lh9998@princeton.edu


# conda activate py3.9-torch
# cd /home/lh9998/exp3/slot-attention-pytorch


# python train_generator.py --dataset=dermamnist --train --evaluate --log --num_epochs=100
python train_generator.py --dataset=dermamnist --evaluate --log --slot_checkpoint=./models_tmp/2023_09_17_21_33_39/SAAutoEncoder_epoch90.pt
# python train.py --dataset=isic2020 --train --evaluate --log --batch_size=32 --num_epochs=400
# python train.py --dataset=isic2020 --evaluate --log --checkpoint=/home/lh9998/exp3/slot-attention-pytorch/models_tmp/2023_09_19_01_33_36/SAAutoEncoder_epoch20.ckpt

python train_classifier.py --dataset=dermamnist --train --log --num_epochs=50
python train_classifier.py --dataset=dermamnist --train --log --learning_rate=0.001 --num_epochs=50 --slot_checkpoint=./models_tmp/2023_09_17_21_33_39/SAAutoEncoder_epoch90.pt
python train_classifier.py --dataset=augmented --dataset_path=/home/lh9998/exp3/slot-attention-pytorch/models_tmp/generator_2023_10_03_03_55_52/ --slot_checkpoint=./models_tmp/2023_09_17_21_33_39/SAAutoEncoder_epoch90.pt --train --log --num_epochs=200 --learning_rate=0.001
python train_classifier.py --dataset=dermamnist --train --log --num_epochs=100 --learning_rate=0.001
