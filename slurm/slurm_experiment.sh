#!/bin/bash
#SBATCH -J SBX-CROSSQ
#SBATCH -a 1-11
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --mem-per-cpu=5000
#SBATCH -t 72:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx3090|a5000'
#SBATCH -o /home/palenicek/projects/sbx-crossq/logs/%A_%a.out.log
#SBATCH -e /home/palenicek/projects/sbx-crossq/logs/%A_%a.err.log
#SBATCH --comment daniel.palenicek@tu-darmstadt.com
## Make sure to create the logs directory /home/user/Documents/projects/prog/logs, BEFORE launching the jobs.

# Setup Env
SCRIPT_PATH=$(dirname $(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}'))
echo $SCRIPT_PATH

source $SCRIPT_PATH/conda_hook
echo "Base Conda: $(which conda)"
eval "$($(which conda) shell.bash hook)"
conda activate crossq
echo "Conda Env:  $(which conda)"

export GTIMER_DISABLE='1'
echo "GTIMER_DISABLE: $GTIMER_DISABLE"

cd $SCRIPT_PATH
echo "Working Directory:  $(pwd)"


python /home/palenicek/projects/sbx/train.py \
    -env $ENV \
    -seed $SLURM_ARRAY_TASK_ID \
    -wandb_mode 'online'
