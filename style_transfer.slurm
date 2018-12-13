#!/bin/bash
#
#   -- Copy/edit this script as desired.  Launch by executing
#      "sbatch knl.serial.slurm" on a Stampede2 login node.
#
#   -- Serial codes run on a single node (upper case N = 1).
#        A serial code ignores the value of lower case n,
#        but slurm needs a plausible value to schedule the job.
#
#   -- For a good way to run multiple serial executables at the
#        same time, execute "module load launcher" followed
#        by "module help launcher".

#----------------------------------------------------

#SBATCH -J style_transfer           # Job name
#SBATCH -o results/all_0004_2.o%j       # Name of stdout output file
#SBATCH -e results/all_0004_2.e%j       # Name of stderr error file
#SBATCH -p gtx             # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=aaron.j.m.chang@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A CS388-NLP	   # Allocation name (req'd if you have more than 1)

# Other commands must follow all #SBATCH directives...

module list
pwd
date

# Launch serial code (add options here)

python main.py --hidden_size 512 --epochs 10 --esv data/esv.txt --kjv data/kjv.csv --lr 0.0004 --train_path data/all/kjv_train.csv --dev_path data/all/kjv_dev.csv --test_path data/all/kjv_test.csv

# ---------------------------------------------------
