#!/bin/bash
source ~/.bashrc
conda activate dmdn
cd /home/ddinosse/GAN_Anomaly_Detection

sleep 1
job=196303
for type in 'fill_side' #'big_order'
do
    for ampl in 1.5  2.0 2.5 3.0 3.5 #4 4.5 5 5.5 6
    do
        python anomaly_detection.py -N=1 -d=3 -j=$job -tg=dense -td=dense -nlg=3 -nld=3 -ls=original -tc=50 --ampl=$ampl -at=$type
    done
done

python anomaly_detection.py -N=1 -d=3 -j=$job -tg=dense -td=dense -nlg=3 -nld=3 -ls=original -tc=50 --ampl=0 -at='liquidity_crisis'