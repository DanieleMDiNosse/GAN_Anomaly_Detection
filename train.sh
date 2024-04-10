#!/bin/bash
source ~/.bashrc
conda activate dmdn
cd /home/ddinosse/GAN_Anomaly_Detection

sleep 0.5
type_gen=dense
type_disc=dense
n_layers_gen=3
n_layers_disc=3
N_days=1
T_condition=1
T_gen=1
loss=original
batch_size=32
depth=5
latent_dim=$(($depth*10))
# skip_connectionis False if not passed, True if passed

python train.py -N=$N_days -bs=$batch_size -d=$depth -ld=$latent_dim -tg=$type_gen -td=$type_disc -nlg=$n_layers_gen -nld=$n_layers_disc -ls=$loss -Tc=$T_condition -Tg=$T_gen --skip_connection 





