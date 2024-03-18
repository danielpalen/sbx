#!/bin/bash

# # 1 # Main Plots #############
# # Fig 1,3,5
timesteps=5000000
q_bias=0

for env in 'Walker2d-v4' 'Hopper-v4' 'Ant-v4' 'HalfCheetah-v4'; do
    ALGO='sac' ENV=$env ACT='relu' LR='0.001' UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048  N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_multi.sh;
done

for env in 'Humanoid-v4' 'HumanoidStandup-v4'; do
    ALGO='sac' ENV=$env ACT='relu' LR='0.001' UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048  N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh;
done
