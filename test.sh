#!/bin/bash
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
cd $HOME/dcdi # <- actual job commands start here
source venv/bin/activate
export PYTHONPATH=./:{PYTHONPATH}
python3 -u ./main.py --train --data-path ./data/small_observational_data/data_p3_e3.0_n20000_nn/ --num-vars 3 --i-dataset 1 --exp-path exp --model DCDI-DSF --lr 0.005 --metrics_path ./test_3.csv

##apptainer exec --writable dcdi_container.sif/ ./test.sh
