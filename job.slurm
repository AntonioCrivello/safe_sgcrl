#!/bin/bash
#SBATCH --job-name=contrastive_gpu                      
#SBATCH --cpus-per-task=1         
#SBATCH --mem-per-cpu=8G               
#SBATCH --gres=gpu:1         
#SBATCH --time=4:00:00                                   
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





# python lp_contrastive.py \
#     --env point_Spiral11x11 \
#     --seed "$SEED" \
#     --goal_neg_actor_steps 5000000 \
#     --num_steps 5000000 \
#     "${hidden_flags[@]}"
  ##  --goal_pos_actor_steps 4000000 \
# --perturbed_negatives_num 5 \
    # --perturbed_negatives_goal_num 5\
    #    --use_residual_mlp \
    #    --goal_pos_frac 0.05 \
        # -goal_pos_actor_steps 8000000 \

        ## top right --region_bounds=0,5:5,11 \
        ## bottom left --region_bounds=5,0:11,5 \
        #--region_bounds=5,3:9,8 \
#       --region_bounds=0,8:3,11 \
      # --nostop_grad_fixed \
      # --region_bounds=7,0:11,4 \ narroew  line in fourrooms
      ## x >> 9 - 10 , y >> 5, 7 >>>>  --region_bounds=9,5:10,7 \ --region_bounds=low_x,low_y:high_x,high_y \
      #      --region_bounds=1,3:4,10 \
            # --nonegative_goal_repr \



python -u lp_contrastive.py \
      --env point_FourRooms\
      --num_steps 8000000 \
      --region_bounds=0,5:5,11 \


exit 0  