#!/bin/bash
source ~/.bashrc
conda activate dmdn
cd /home/ddinosse/GAN_Anomaly_Detection

sleep 0.5
data=ar1
type_gen=dense
type_disc=dense
n_layers_gen=2
n_layers_disc=2
T_condition=1
T_gen=1
loss=original_fm
batch_size=32
depth=2
latent_dim=$(($depth*25))
n_epochs=10000
# skip_connectionis False if not passed, True if passed
# conditional should be passed if you are training a conditional GAN
# synthetic should be passed if you are testing on synthetic data
# clipping should be passed if you want to clip the gradients of the discriminator

python train_synthetic.py -d=$data --n_epochs=$n_epochs -bs=$batch_size -ld=$latent_dim -tg=$type_gen -td=$type_disc -nlg=$n_layers_gen -nld=$n_layers_disc -ls=$loss -Tc=$T_condition -Tg=$T_gen  --clipping --skip_connection --synthetic





