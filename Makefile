check:
	squeue -u smith

gpu:
	srun --jobid=$(JOBID) --pty nvitop

cpu:
	srun --jobid=$(JOBID) --pty htop

clear-logs:
	rm -rf logs/*.out

# slurm needs logs/ to exist before it can open the job's -o file
logs:
	mkdir -p logs

train: logs
	sbatch scripts/train.slrm

train-array: logs
	sbatch scripts/train_array.slrm

eval: logs
	sbatch scripts/eval.slrm

eval-geometries: logs
	sbatch scripts/eval_geometries.slrm
