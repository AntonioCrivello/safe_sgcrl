#!/bin/bash
#SBATCH --job-name=contrastive_gpu     
#SBATCH --array=0-15%16                          
#SBATCH --cpus-per-task=1         
#SBATCH --mem-per-cpu=8G               
#SBATCH --gres=gpu:1         
#SBATCH --time=00:59:00                                   
#SBATCH --mail-user=mb6458@princeton.edu
#SBATCH --account=eysenbach

module purge
module load anaconda3/2023.3
module load cudatoolkit/11.3 cudnn/cuda-11.x/8.2.0

# Activate the correct environment
source /home/mb6458/miniconda3/bin/activate contrastive_rl

# Set the library paths
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/lib64
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mb6458/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia


# *** NEW: Adjust JAX GPU memory preallocation ***
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
# Debug prints
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_HOME: $CUDA_HOME"
nvidia-smi



seeds=(308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}





python -m experiments.similarity_posterior_exp \
  --alpha 0.1 \
  --env_name point_Impossible \
  --seed $SEED \
  --log_dir logs \
  --alg contrastive_cpc \
  --ckpt_list 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 30 35 40\
  --NUM_AXES 2 \
  --NUM_EPISODES 5 \
  --plot_psi \
  --action_mode "actor_sample"\

exit 0



