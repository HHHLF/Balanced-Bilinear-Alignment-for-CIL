#!/bin/bash  
for((i=0;i<=9;i++));  
do   
    # echo $i
    TXT_PATH='/data/zyx/zyx1/GMM-gpm1-ema/results/20250509133/test_'"$i"'.txt'
    CKPT_PATH='/data/zyx/zyx1/GMM-gpm1-ema/minigpt4/output2/20250509133/'"$i"'/checkpoint_4.pth'
    python batch_eval.py --cfg-path eval_configs/minigpt4_eval_all_tasks_imgr.yaml \
    --gpu-id 1 --task-id $i --txt-path $TXT_PATH \
    --ckpt-path $CKPT_PATH
done  
# python get_score_all.py