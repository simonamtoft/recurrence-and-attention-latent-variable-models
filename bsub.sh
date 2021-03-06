#!/bin/sh
#BSUB -q gpua100
#BSUB -J "TrainDRAW"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 23:59
#BSUB -R "rusage[mem=8GB]"
##BSUB -B
### -- send notification at completion--
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo gpu-%J.out
#BSUB -eo gpu-%J.err
# -- end of LSF options --

# Load the cuda module
module load cuda/11.1
module load cudnn/v8.0.4.30-prod-cuda-11.1

source venv/bin/activate
# python run_vae.py
# python run_lvae.py
python run_draw.py
# python compute_test.py
