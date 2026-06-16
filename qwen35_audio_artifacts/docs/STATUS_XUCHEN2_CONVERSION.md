# xuchen2 Qwen3-Omni MoE 转换与训练状态

## 日期
2026-06-03

## 任务目标
将 `/data/xuchen2/model/Qwen3-Omni-30B-A3B-Captioner` (MoE 模型) 转换为 MindSpeed MCore 格式，并使用完全对齐 hulk 的配置进行 LoRA 微调训练。

---

## ✅ 已完成的工作

### 1. 权重抽取 (HF Omni → 标准 HF MoE)
**状态**: ✅ 成功

- **输入**: `/data/xuchen2/model/Qwen3-Omni-30B-A3B-Captioner` (嵌套 config，含 audio/visual 编码器)
- **输出**: `/data/sejin/models/Qwen3-Omni-30B-A3B-text-extracted` (已删除以释放空间)
- **脚本**: `/data/sejin/baseline_26/scripts/extract_thinker_text.py`
- **结果**: 
  - 抽取了 `thinker.model.*` 和 `thinker.lm_head` (18867 个权重)
  - 去除 audio_tower/visual 模态编码器
  - 平铺 config 为标准 `Qwen3MoeConfig` (model_type=qwen3_moe)
  - 15 个 safetensors 分片，共 61GB

### 2. 权重转换 (HF MoE → MindSpeed MCore)
**状态**: ✅ 成功

- **输入**: 抽取后的标准 HF 格式
- **输出**: `/data/sejin/models/Qwen3-Omni-30B-A3B-hulk_tp1_pp1_ep8` (77GB)
- **脚本**: `/data/sejin/baseline_26/scripts/convert_xuchen2_qwen3omni_moe.sh`
- **并行配置**: TP=1, PP=1, EP=8 (对齐 hulk)
- **耗时**: 962 秒 (~16 分钟)
- **验证**: 
  - 8 个 EP rank (mp_rank_00_000 ~ mp_rank_00_007)
  - 每个 rank 9.7GB
  - grouped GEMM 格式: `weight1` [2048, 24576], `weight2` [12288, 2048]
  - 保存的 args 正确: num_layers=48, num_experts=128, EP=8, TP=1, PP=1

### 3. 数据预处理
**状态**: ✅ 成功

- **输入**: `/data/sejin/baseline_26/data_hulk_dist_30k/train.jsonl` (30000 条 OpenAI messages 格式)
- **输出**: `/data/sejin/baseline_26/data_hulk_dist_30k/qwen3_sft_packed_*_document.{bin,idx}`
- **脚本**: `/data/sejin/baseline_26/scripts/preprocess_hulk_data.sh`
- **配置**:
  - SharegptStyleInstructionHandler
  - seq_length=8192, pack=true
  - qwen3 prompt 模板 (ChatML + reasoning)
- **结果**: 30000/30000 valid (100%), 打包为 2318 个 8192-token 样本

### 4. 训练脚本创建
**状态**: ✅ 完成 (脚本已创建并配置正确)

- **脚本**: `/data/sejin/baseline_26/scripts/train_xuchen2_hulk_aligned.sh`
- **配置 (完全对齐 hulk)**:
  - 并行: TP=1, PP=1, EP=8, CP=2 (Ulysses)
  - LoRA: r=32, alpha=64, dropout=0.1, target=qkv+proj
  - 序列长度: 8192
  - 超参: lr=5e-6, clip=5.0, warmup=0.0
  - 不使用 swap-optimizer
- **数据路径**: `/data/sejin/baseline_26/data_hulk_dist_30k/qwen3_sft`
- **权重路径**: `/data/sejin/models/Qwen3-Omni-30B-A3B-hulk_tp1_pp1_ep8`
- **Tokenizer**: `/data/sejin/models/Qwen3-30B-A3B-Base` (与 Captioner 编码一致)

---

## ❌ 遇到的问题

### 问题：训练启动失败 (多次尝试)

#### 初始错误 (已解决)
```
ModuleNotFoundError: No module named 'mindspeed.features_manager.features_manager'
```

**原因**: venv 中存在 stale editable installs:
- `mindspeed-0.8.0` (editable) → `/data/xuchen2/git/qwen3-mindspeed/third_party/mindspeed`
- `mindspeed_llm-0.0.1` (editable) → `/data/xuchen2/git/qwen3-mindspeed/third_party/mindspeed-llm`

这些 editable 通过 `__editable___*.pth` 注册 MetaPathFinder，**优先级高于 PYTHONPATH**。旧版本 (0.8.0) 缺少 `features_manager/features_manager.py` 模块。

**解决方案**: 
```bash
pip uninstall -y mindspeed mindspeed_llm
```

卸载后，验证 import 走正确路径:
```python
import mindspeed  # → mindspeed-core-26.0.0/mindspeed/__init__.py ✓
from mindspeed.features_manager.features_manager import MindSpeedFeaturesManager  # ✓
```

#### 当前问题：后台进程无法持久化

**现象**: 
- 使用 harness 的 `run_in_background=true` 模式启动后，进程立即终止
- 使用 `nohup ... &` 在多行脚本中启动，zsh glob 错误导致整个命令中止
- 使用单行 `nohup ... &` 启动成功 (pid 1947037)，但检查时发现进程已不存在

**根本原因 (推测)**:
1. **Harness 工具调用生命周期限制**: `Bash` 工具的后台任务可能在工具调用返回后被清理
2. **zsh glob 中止**: 多行命令中的 `*.log` glob 如果无匹配，zsh 默认报错并中止 (需要 `setopt NULL_GLOB`)
3. **进程组管理**: torchrun 启动的 8 个 worker 进程可能因为某个环境问题 (PYTHONPATH 继承、端口冲突等) 迅速失败退出

**已尝试的启动方式** (均未成功持久化):
- `nohup bash script.sh > log 2>&1 &` (后台模式)
- `setsid bash script.sh ...` (新会话)
- `Bash` 工具的 `run_in_background: true`
- `env ITERS=20 ... bash script.sh` (前台，timeout 测试)

---

## 📋 待解决步骤

### 选项 A: 直接在终端运行 (绕过 harness)

让用户自己在终端执行:
```bash
cd /data/sejin/third_party/mindspeed-llm-26.0.0
nohup bash /data/sejin/baseline_26/scripts/train_xuchen2_hulk_aligned.sh \
  > /data/sejin/baseline_26/logs/xuchen2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
disown
```

**优点**: 完全脱离 AI agent 的进程管理限制  
**缺点**: 需要用户手动操作

### 选项 B: 调试并修复环境问题

1. **验证 torchrun 单进程启动**:
   ```bash
   cd /data/sejin/third_party/mindspeed-llm-26.0.0
   source /data/sejin/baseline_26/scripts/env_cann85.sh
   ITERS=1 /data/sejin/env/venv_26b/bin/torchrun --nproc_per_node 1 \
     --master_port 6030 posttrain_gpt.py <all-args>
   ```
   查看是否能加载权重、初始化模型

2. **逐步扩展到 8 进程**:
   - 单进程成功 → 2 进程 → 4 进程 → 8 进程
   - 每步验证 PYTHONPATH 继承、端口分配、集合通信初始化

3. **检查日志文件**:
   - 如果之前启动过但立即退出，查看是否有残留日志
   - `/data/sejin/baseline_26/logs/xuchen2_wrap_*.log`
   - `/data/sejin/baseline_26/logs/xuchen2_smoke_*.log`

### 选项 C: 使用 systemd/tmux session (推荐)

创建一个 tmux session 运行训练:
```bash
tmux new-session -d -s xuchen2_train
tmux send-keys -t xuchen2_train "cd /data/sejin/third_party/mindspeed-llm-26.0.0" C-m
tmux send-keys -t xuchen2_train "source /data/sejin/baseline_26/scripts/env_cann85.sh" C-m
tmux send-keys -t xuchen2_train "bash /data/sejin/baseline_26/scripts/train_xuchen2_hulk_aligned.sh 2>&1 | tee /data/sejin/baseline_26/logs/xuchen2_tmux_\$(date +%Y%m%d_%H%M%S).log" C-m
```

监控:
```bash
tmux attach -t xuchen2_train
# Ctrl+B D 分离
```

---

## 🔍 验证检查清单

在启动训练前，确认以下项:

- [x] 权重转换完成 (`/data/sejin/models/Qwen3-Omni-30B-A3B-hulk_tp1_pp1_ep8/iter_0000001/mp_rank_00_*/`)
- [x] 数据预处理完成 (`qwen3_sft_packed_*_document.{bin,idx}`)
- [x] Tokenizer 可用 (`/data/sejin/models/Qwen3-30B-A3B-Base/vocab.json`)
- [x] PYTHONPATH 包含 mindspeed-core-26.0.0
- [x] `import mindspeed` 解析到 26.0.0 (不是 0.8.0)
- [x] NPU 8 卡空闲 (HBM ~3-4GB 残留)
- [x] 磁盘空间充足 (63GB 可用)
- [ ] **训练进程成功启动并运行**

---

## 📁 关键文件位置

### 脚本
- 抽取: `/data/sejin/baseline_26/scripts/extract_thinker_text.py`
- 转换: `/data/sejin/baseline_26/scripts/convert_xuchen2_qwen3omni_moe.sh`
- 预处理: `/data/sejin/baseline_26/scripts/preprocess_hulk_data.sh`
- 训练: `/data/sejin/baseline_26/scripts/train_xuchen2_hulk_aligned.sh`
- 环境: `/data/sejin/baseline_26/scripts/env_cann85.sh`

### 数据
- 权重 (MCore): `/data/sejin/models/Qwen3-Omni-30B-A3B-hulk_tp1_pp1_ep8/` (77GB)
- 训练数据: `/data/sejin/baseline_26/data_hulk_dist_30k/qwen3_sft_packed_*` (73MB × 3)
- Tokenizer: `/data/sejin/models/Qwen3-30B-A3B-Base/`

### 日志
- 转换日志: `/data/sejin/baseline_26/logs/convert_20260603_193030.log`
- 预处理日志: `/data/sejin/baseline_26/logs/preprocess_20260603_200556.log`
- 训练日志 (预期): `/data/sejin/baseline_26/logs/xuchen2_hulk_*.log`

### 文档
- hulk 配置对比: `/data/sejin/baseline_26/reports/HULK_VS_BASELINE_COMPARISON.md`
- 转换指南: `/data/sejin/baseline_26/README_XUCHEN2_CONVERSION.md`

---

## 💡 下一步建议

1. **让用户手动启动训练** (最快):
   ```bash
   # 在终端执行
   cd /data/sejin/third_party/mindspeed-llm-26.0.0
   source /data/sejin/baseline_26/scripts/env_cann85.sh
   bash /data/sejin/baseline_26/scripts/train_xuchen2_hulk_aligned.sh 2>&1 | tee /data/sejin/baseline_26/logs/xuchen2_manual.log
   ```

2. **或使用 tmux** (推荐):
   ```bash
   tmux new -s train
   cd /data/sejin/third_party/mindspeed-llm-26.0.0
   source /data/sejin/baseline_26/scripts/env_cann85.sh
   bash /data/sejin/baseline_26/scripts/train_xuchen2_hulk_aligned.sh
   # Ctrl+B D 分离，后台运行
   ```

3. **监控训练**:
   ```bash
   # 查看实时日志
   tail -f /data/sejin/baseline_26/logs/xuchen2_*.log | grep -E "iteration|loss|samples/sec"
   
   # 查看 NPU 利用率
   watch -n 5 npu-smi info
   
   # 查看进程
   pgrep -af posttrain_gpt
   ```

---

## ⚠️ 已知限制

1. **磁盘空间紧张**: 63GB 可用 (权重 77GB + 数据 0.2GB + checkpoint 预留 ~10GB)
2. **CP=2 (Ulysses) 未验证**: MindSpeed 26.0.0 + CANN 8.5 对 Ulysses 的支持可能有问题，如遇错误可临时回退 CP=1/EP=4
3. **swap-optimizer 未启用**: hulk 用纯 GPU ZeRO-2，但 MindSpeed 配置为 `--use-distributed-optimizer` (ZeRO-1 等效)
4. **数据分布**: 当前用 30k 测试数据 (mean ~614 tokens)，与 hulk 原始数据 (382k 样本) 规模不同

---

**创建时间**: 2026-06-03 20:30  
**最后更新**: 2026-06-03 21:15  
**状态**: 训练脚本就绪，等待用户手动启动
