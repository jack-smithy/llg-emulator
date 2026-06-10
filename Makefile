train:
	sbatch run.slrm

check:
	squeue -u js82197

gpu:
	srun --jobid=$(JOBID) --pty uvx nvitop