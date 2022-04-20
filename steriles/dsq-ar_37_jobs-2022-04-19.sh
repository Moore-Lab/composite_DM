#!/bin/bash
#SBATCH --output dsq-ar_37_jobs-%A_%1a-%N.out
#SBATCH --array 0-9%2000
#SBATCH --job-name dsq-ar_37_jobs
#SBATCH --mem-per-cpu 8g -t 30:00 --mail-type NONE

# DO NOT EDIT LINE BELOW
/gpfs/loomis/apps/avx/software/dSQ/1.05/dSQBatch.py --job-file /vast/palmer/home.grace/dcm42/impulse/steriles/job_files/ar_37_jobs.txt --status-dir /vast/palmer/home.grace/dcm42/impulse/steriles

