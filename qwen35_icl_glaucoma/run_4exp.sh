#!/usr/bin/env bash
# Autonomous 4-experiment runner: train 4 x 27B LoRA in parallel (2 GPUs each),
# wait for all, then eval each on the 120-test, then write a combined summary.
# Self-contained / detached (run via setsid+nohup) so it survives ssh drops.
#
#  exp1: quadmix data, borderline x1.2 (renorm to mean=1)   -> GPU 0,1
#  exp2: quadmix data, borderline x1.5 (renorm)             -> GPU 2,3
#  exp3: quadmix data, borderline x2.0 (renorm)             -> GPU 4,5
#  exp4: v2_cw data MINUS 25 forced cases (bl x1.0)         -> GPU 6,7
set -u
source /data/home/gracechen/miniforge3/etc/profile.d/conda.sh
conda activate llm
cd /data/home/gracechen/qwen35_icl_glaucoma
SNAP=/data/home/gracechen/.cache/huggingface/hub/models--google--medgemma-27b-it/snapshots/2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f
export COT_EYES_ROOT=/data/home/gracechen/COT_Eyes
mkdir -p outputs/exp4_logs
COMMON="--model-id $SNAP --class-weighted --oversample 1 --lora-r 8 --lora-alpha 16 --epochs 8 --lr 1e-4 --batch-size 1 --grad-accum 8"
QM=DATA/sft_train_tc_quadmix.jsonl
V2=DATA/sft_train_tc.jsonl

echo "==== [$(date)] TRAINING START ===="
CUDA_VISIBLE_DEVICES=0,1 python -u train_medgemma27b_tc.py $COMMON --data $QM \
  --bl-mult 1.2 --bl-renorm --out outputs/exp1_qm_bl12 \
  > outputs/exp4_logs/train_exp1.log 2>&1 & P1=$!
sleep 40
CUDA_VISIBLE_DEVICES=2,3 python -u train_medgemma27b_tc.py $COMMON --data $QM \
  --bl-mult 1.5 --bl-renorm --out outputs/exp2_qm_bl15 \
  > outputs/exp4_logs/train_exp2.log 2>&1 & P2=$!
sleep 40
CUDA_VISIBLE_DEVICES=4,5 python -u train_medgemma27b_tc.py $COMMON --data $QM \
  --bl-mult 2.0 --bl-renorm --out outputs/exp3_qm_bl20 \
  > outputs/exp4_logs/train_exp3.log 2>&1 & P3=$!
sleep 40
CUDA_VISIBLE_DEVICES=6,7 python -u train_medgemma27b_tc.py $COMMON --data $V2 \
  --bl-mult 1.0 --exclude-ids DATA/forced_ids.txt --out outputs/exp4_v2cw_drop25 \
  > outputs/exp4_logs/train_exp4.log 2>&1 & P4=$!

wait $P1 $P2 $P3 $P4
echo "==== [$(date)] ALL TRAINING DONE -> EVAL ===="

# eval each (27B single-process, device_map across its 2 GPUs, all 120 cases)
EV="python -u eval_medgemma27b_tc.py --base $SNAP"
CUDA_VISIBLE_DEVICES=0,1 $EV --adapter outputs/exp1_qm_bl12 --indicators DATA/predicted_indicators_test_quadmix.json   --out outputs/eval_exp1 > outputs/exp4_logs/eval_exp1.log 2>&1 & E1=$!
CUDA_VISIBLE_DEVICES=2,3 $EV --adapter outputs/exp2_qm_bl15 --indicators DATA/predicted_indicators_test_quadmix.json   --out outputs/eval_exp2 > outputs/exp4_logs/eval_exp2.log 2>&1 & E2=$!
CUDA_VISIBLE_DEVICES=4,5 $EV --adapter outputs/exp3_qm_bl20 --indicators DATA/predicted_indicators_test_quadmix.json   --out outputs/eval_exp3 > outputs/exp4_logs/eval_exp3.log 2>&1 & E3=$!
CUDA_VISIBLE_DEVICES=6,7 $EV --adapter outputs/exp4_v2cw_drop25 --indicators DATA/predicted_indicators_test_newheads.json --out outputs/eval_exp4 > outputs/exp4_logs/eval_exp4.log 2>&1 & E4=$!
wait $E1 $E2 $E3 $E4
echo "==== [$(date)] ALL EVAL DONE ===="

python3 - <<'PY'
import json
labels={'eval_exp1':'exp1 quadmix bl1.2','eval_exp2':'exp2 quadmix bl1.5','eval_exp3':'exp3 quadmix bl2.0','eval_exp4':'exp4 v2_cw drop25'}
out={}
for d,name in labels.items():
    try:
        s=json.load(open(f'outputs/{d}/summary.json'))
        out[name]={'acc':s.get('acc'),'per_class':{k:v.get('recall') for k,v in s.get('per_class',{}).items()}}
    except Exception as e:
        out[name]={'error':str(e)}
json.dump(out,open('outputs/EXP4_RESULTS.json','w'),indent=2,ensure_ascii=False)
print(json.dumps(out,indent=2,ensure_ascii=False))
PY
echo "==== [$(date)] WROTE outputs/EXP4_RESULTS.json ===="
