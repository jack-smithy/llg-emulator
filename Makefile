check:
	squeue -u smith

gpu:
	srun --jobid=$(JOBID) --pty nvitop

cpu:
	srun --jobid=$(JOBID) --pty htop

clear-logs:
	rm -rf logs/*.out

train:
	sbatch scripts/train.slrm

train-array:
	sbatch scripts/train_array.slrm

eval:
	sbatch scripts/eval.slrm

eval-geometries:
	sbatch scripts/eval_geometries.slrm
