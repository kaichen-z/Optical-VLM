# GitHub Push List — 训练相关代码(产出 0.708 best result 的 pipeline)

两段式:**阶段1** RETFound 头预测临床指标 → **阶段2** MedGemma-27B LoRA 做 CoT 诊断。
本列表只收**训练相关 code**;消融/分析/旧路线代码默认排除(见 §C),混在一起的见 §D。
最优结果 exp4 (= v2_cw 数据去掉 25 forced) = **ACC 0.7083**。

---

## A. 阶段1 — `retfound_header_finetune/`(RETFound 指标头训练)

### A1. 公共依赖(必须,所有 trainer 都 import)
```
retfound_header_finetune/config.py            # RETFound 路径/架构配置
retfound_header_finetune/retfound_backbone.py # 冻结 ViT-L 加载 (load_retfound)
retfound_header_finetune/heads.py             # 头部 FC (linear/mlp)
retfound_header_finetune/baseline.py          # pooling: mean
retfound_header_finetune/tier1.py             # pooling
retfound_header_finetune/tier2a.py            # pooling (门控注意力)
retfound_header_finetune/tier2b.py            # pooling (单query注意力)
retfound_header_finetune/tier3.py             # pooling (多头注意力, exp4 rim 头用)
retfound_header_finetune/make_folds.py        # 5-fold 划分 (OOF 用)
```

### A2. ★ 关键路径 trainer(exp4 的 3 个指标头就靠这 3 个)
```
retfound_header_finetune/train.py             # ★ numeric CDR 回归头 (tier2a_linear)
retfound_header_finetune/train_order_abn.py   # ★ ISNT order+象限异常 联合头 (orderabn_tier3)
retfound_header_finetune/train_signs.py       # ★ 6 个 glaucomatous signs 头
retfound_header_finetune/dataset.py           #   train.py 的数据加载
retfound_header_finetune/dataset_order_abn.py #   train_order_abn 的数据
retfound_header_finetune/dataset_signs.py     #   train_signs 的数据
```

### A3. ★ 指标生成(把头的输出拼成 stage2 吃的 JSON)
```
retfound_header_finetune/assemble_oof.py            # ★ 生成 predicted_indicators_*_newheads.json (exp4 用)
retfound_header_finetune/predict_heads_train.py     #   头在训练集上预测(OOF)
retfound_header_finetune/predict_indicators_imagelist.py
retfound_header_finetune/build_predicted_indicators.py
```

### A4. 其他头 trainer(训练code,但**不在 0.708 路径**上;算 stage1 内的探索,自己决定收不收)
```
train_isnt.py / dataset_isnt.py            # numeric ISNT 回归
train_order.py                             # 纯 order 排序头
train_cdr_verdict.py / dataset_cdr_verdict.py  # CDR 分类(verdict)头
train_tri_head.py / dataset_tri.py         # 共享pooling 三任务联合头
train_quad_head.py / train_quad_reg_head.py / dataset_quad.py / dataset_quadreg.py  # 四指标共享pooling头(quadmix/quadreg)
train_laterality.py / dataset_laterality.py    # OD/OS 左右眼头
train_rim_flip.py / train_rim_multi.py / dataset_rim_flip.py / dataset_rim_multi.py
assemble_cdr_verdict_oof.py / assemble_quad_mixed.py / assemble_quadreg_mixed.py / predict_quad.py / predict_quadreg.py / predict_train_insample.py
```

---

## B. 阶段2 — `qwen35_icl_glaucoma/`(MedGemma-27B LoRA,★主结果)

### B1. ★ 核心训练
```
qwen35_icl_glaucoma/train_medgemma27b_tc.py   # ★ LoRA 训练引擎(自包含,无本地依赖)
qwen35_icl_glaucoma/assemble_sft_tc.py        # ★ 生成 sft_train_tc.jsonl (v2_cw CoT 训练数据)
qwen35_icl_glaucoma/run_4exp.sh               # ⚠️ exp4 编排在这(见 §D,混了 exp1-3 消融)
```

### B2. 评测(拿到 0.708 这个数必须有;用户说"训练相关",但没它复现不了结果 → 建议收)
```
qwen35_icl_glaucoma/eval_medgemma27b_tc.py    # exp4 评测(HF)
qwen35_icl_glaucoma/run_qwen35_incontext.py   # ⚠️ 必须收:eval 依赖它的 parse_diagnosis/load_test_records(虽是旧Qwen路线,但提供工具函数)
```

### B3. 工程文件
```
qwen35_icl_glaucoma/requirements.txt
qwen35_icl_glaucoma/README.md
qwen35_icl_glaucoma/push.sh / pull.sh          # 本地↔server 同步(可选)
```

---

## C. 排除(消融/旧路线/分析,默认不 push)
```
# 阶段2 消融:
build_imageonly_data.py, eval_imageonly_lora.py, run_imageonly.sh   # 纯图基线
eval_zeroshot_imageonly.py, eval_zeroshot_imageonly_vllm.py         # zero-shot floor
run_bigbatch16.sh                                                   # 大batch实验
assemble_sft_tc_quadmix.py / _quadreg.py / _insample.py            # 消融数据变体
eval_medgemma27b_tc_vllm_calibV1/V2/V3.py, *_diagfirst.py          # 校准/排序消融
eval_refuge.py, eval_refuge_vllm.py                                # REFUGE 跨数据集
run_medgemma_cdr_level.py, run_medgemma_cdr_qual*.py, run_img_plus_indicators.py
run_textcot_oracle.py, sft_qwen_textcot.py                          # 文本/oracle 消融
build_sft_data.py, build_sft_data_pred.py                          # 旧数据构建
run_full.sh, run_medgemma_indicators.py                            # 旧 Qwen35 路线
# 分析/画图:
analyze_imgind.py, analyze_results.py, compute_metrics.py, make_confmat.py,
plot_cot_ablation_cm.py, merge_shards.py, eval_sft_predicted.py, test_smoke.sh
```

---

## D. ⚠️ 混在一起的(收了,但要知道里面有消融)
- **`train_medgemma27b_tc.py`** — 同一个训练引擎,主结果(v2_cw/exp4)和消融(imageonly/bigbatch/quadreg/insample)**都用它**,差别只在命令行参数,不在文件本身。必收。
- **`run_4exp.sh`** — exp4(★最优)和 exp1/2/3(quadmix borderline 倍率消融)写在**同一个脚本**里。收它=连消融一起带进来,无法只切 exp4 那段(除非手动删 exp1-3)。
- **`run_qwen35_incontext.py`** — 旧 Qwen 在上下文路线(已弃用),但 `eval_medgemma27b_tc.py` 依赖它的工具函数,所以**不得不收**。

---

## E. 不进 repo(数据/权重/产出 → `.gitignore`)
```
retfound_weights/*.pth            # RETFound backbone 3.95G
*/outputs/                        # 训练 adapter + 评测结果 + 头权重
*/DATA/*.jsonl, */DATA/*.json     # SFT 数据/指标(大,且可由 code 重生成)
*/DATA/test_120/, COT_Eyes/       # 图片
*.png, *.html                     # 图/可视化
__pycache__/
```
> exp4 权重(`outputs/exp4_v2cw_drop25/adapter_model.safetensors` 70M)若要进 repo 用 **Git LFS**;否则只放 code,权重单独存。
