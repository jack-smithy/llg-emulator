CONFIG ?= configs/sil-wide-long.yaml
SEED ?= 0
LINKS ?= 3

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

# two seeds of a short recipe: make train CONFIG=configs/aeff-tendency.yaml
train: logs
	sbatch --array=0-1 scripts/train.slrm $(CONFIG)

# one seed of a long recipe as a chain of 3 h links: make chain CONFIG=... SEED=0 LINKS=3
chain: logs
	scripts/chain.sh $(CONFIG) $(SEED) $(LINKS) --gres=gpu:full:1 -c 8
