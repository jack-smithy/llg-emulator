check:
	squeue -u smith

gpu:
	srun --jobid=$(JOBID) --pty nvitop

clear-logs:
	rm -rf logs/*.out

train:
	sbatch scripts/train.slrm

train-array:
	sbatch scripts/train_array.slrm

eval:
	sbatch scripts/eval.slrm
