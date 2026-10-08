CONFIG ?= configs/sil-wide-long.yaml

check:
	squeue -u smith

gpu:
	srun --jobid=$(JOBID) --pty nvitop

cpu:
	srun --jobid=$(JOBID) --pty htop

# slurm needs logs/ to exist before it can open the job's -o file
logs:
	mkdir -p logs

clear-logs:
	rm -rf logs/*.out

# two seeds of a recipe: make train CONFIG=configs/k1-wide-long.yaml
train: logs
	sbatch --array=0-1 scripts/train.slrm $(CONFIG)
