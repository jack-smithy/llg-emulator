train:
	sbatch scripts/run.slrm

eval:
	sbatch scripts/eval.slrm

check:
	squeue -u js82197

gpu:
	srun --jobid=$(JOBID) --pty uvx nvitop