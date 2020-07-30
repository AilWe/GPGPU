Running step:

1. apply for palmetto node:
$ qsub -I -l select=1:ncpus=1:ngpus=1:gpu_model=k20:mem=8gb,walltime=4:00:00

2. add cuda module:
$ module add cuda-toolkit/8.0.44

3. compile
$ ./oclc openCL_mmul

4. run
$ ./openCL_mmul

Result plot:
Code in .ipynb file (run_time_plot.ipynb)
Pdf version in run_time_plot.pdf