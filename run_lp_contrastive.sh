#!/bin/bash
# run_lp_contrastive.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate contrastive_rl            # ➜ env: contrastive_rl_nn
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
echo "$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"




seeds=(31)
devices=(0)

for idx in "${!seeds[@]}"; do
  SEED=${seeds[$idx]}
  DEV=${devices[$idx]}
  LOG="lp_contrastive_seed${SEED}.out"

  echo "▶ Launching seed $SEED on GPU $DEV  →  $LOG"
    CUDA_VISIBLE_DEVICES=$DEV \
    nohup python -u lp_contrastive.py \
      --env point_FourRooms \
      --seed "$SEED" \
      --region_bounds=0,5:5,11 \
      > "$LOG" 2>&1 &   # redirection now belongs to the nohup command
done


echo "✅ All jobs launched (logs in lp_contrastive_seed*.out)"

