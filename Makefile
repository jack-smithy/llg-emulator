check:
	squeue -u smith

gpu:
	srun --jobid=$(JOBID) --pty nvitop

clear-logs:
	rm -rf logs/*.out