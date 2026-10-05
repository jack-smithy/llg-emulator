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

train-sil: logs
	sbatch scripts/train_sil_array.slrm

# the learned-closure LLG (model.ClosureEmulator), two seeds
train-closure: logs
	sbatch scripts/train_closure_array.slrm

# the demag-input model with 1 and 4 frames of context
train-demag: logs
	sbatch --export=ALL,IN_FRAMES=1 scripts/train_demag_array.slrm
	sbatch --export=ALL,IN_FRAMES=4 scripts/train_demag_array.slrm

eval: logs
	sbatch scripts/eval.slrm

eval-geometries: logs
	sbatch scripts/eval_geometries.slrm

# coarse micromagnetics breakdown sweep (datagen/coarse_breakdown.py), one job per film
coarse-breakdown: logs
	FILM=sq256 sbatch -t 3:00:00 scripts/coarse_breakdown.slrm
	FILM=sq512 sbatch -t 5:00:00 scripts/coarse_breakdown.slrm
