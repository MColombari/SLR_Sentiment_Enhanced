To properly run all the bash file in this folder the user needs to be in the "SSTCN" folder.
so for example to run a generic "run.sh" file in the AimageLab server, the user should type:


        sbatch run_file/run.sh


In the "run_output" folder all the output of the scheduled process are visible.


Note how all the path for the dataset are relative to the one in the server, change them accordingly if the repo is running local.