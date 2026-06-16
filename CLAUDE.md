# 项目约束文档 CLAUDE.md
## 一、项目总纲
### 1.项目信息
项目：CANN8.5 + MindSpeed-LLM-26.0.0 对 Qwen3-30B-A3B LoRA 微调性能优化&框架对标评测
硬件：昇腾910B
原有环境：机器预装CANN8.1，**本机已部署完成CANN8.5.0，无需安装包、不用重装驱动，仅环境变量切换使用；项目全程禁用CANN8.1**
模型固定路径：`/data/sejin/models/Qwen3-30B-A3B-Base`
MindSpeed源码固定分支：`https://gitcode.com/Ascend/MindSpeed-LLM/tree/26.0.0`
测试数据参考：`/data/sejin/data`

### 2.项目目标
1. 原始配置，从0开始搭建环境，跑通LoRA微调，占满显存，采集基线全套性能指标；
2. AI Core利用率优化至≥70%；
3. 依托MindSpeed内置Auto Tuning自动优选超参、算子开关，框架自主遍历配置生成最优参数；
4. 输出基线优化分析文档、多轮迭代性能报表、最终完整版汇总评测报告，参考开源&工业界LLM评测规范。
5. 显存要求在50G以上，最好55-60G之间，基本打满。

## 二、环境强制规则
所有编译、安装、训练、调试操作，**必须优先执行下面环境脚本，固定环境，禁止使用系统默认CANN8.1**
```bash
bash
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
# 内部执行训练、调试命令
exit

## 三、硬性约束条款
1. **模型约束**：严禁改动 Qwen3-30B-A3B 原生网络结构、专家数量、MoE 路由规则；仅允许调整训练超参、并行策略、NPU 算子配置、混合精度、LoRA 超参、缓存、通信参数。要验证改动前后的数学一致性，要求必须是数学上一致的。
2. **故障处置**：训练崩溃、OOM、编译报错、算子异常、loss 异常、进程挂死等所有问题，自主定位根因、修复解决，无需反复询问用户，修复完毕继续后续优化步骤。
3. **执行顺序固定不可跳步**
    1）原始默认配置启动 LoRA 训练，占满显存，采集全量基线指标；
    2）结合昇腾官方文档、开源业界最佳实践，输出《基线性能瓶颈分析 & 优化方案文档》；
    3）逐项落地优化项，每轮优化完成输出单轮性能对比报表；
    4）基于报表迭代优化方案，循环调优直至全部可优化项验证完毕；
    5）汇总全部数据输出最终完整版项目性能报告。

## 四、固定资源路径
- MindSpeed-LLM：固定 26.0.0 分支，不随意切版本
- 权重目录：`/data/sejin/models/Qwen3-30B-A3B-Base`
- CANN8.5 环境：
  - CANN：`/usr/local/Ascend/cann-8.5.0/set_env.sh`
  - ATB/NNAL：`/usr/local/Ascend/nnal/atb/set_env.sh`

## 五、优化全覆盖范围
- MindSpeed 内置 Auto Tuning 自动参数调优，依靠框架原生能力自动择优配置；
- 分布式：TP/PP/DP/CP 并行策略、Ring/Ulysses 上下文并行、MoE 专家并行优化；
- NPU 算子：ATB 算子融合、昇腾 FA、Norm 融合、GEMM 优化、KV Cache 优化；
- 显存优化：重计算、梯度累积、混合精度（BF16/FP16/FP8）、HBM 缓存配置；
- HCCL 通信参数、分片优化；
- LoRA 专属：lora-r、学习率、优化器、梯度裁剪等微调参数优选。

## 六、性能指标采集规范（报表必填）
- 吞吐：WPS、TPS、单步耗时、单轮耗时
- 硬件：平均 AI Core 利用率、峰值 AI Core、HBM 带宽、HBM 占用、整机功耗、显存占用率
- 训练：Loss 收敛、无 NaN / 梯度爆炸
- 备注：本轮并行配置、算子开关、AutoTuning 最优参数

## 七、文档规范
- 优化分析文档：项目背景→基线数据→瓶颈分类（算力 / 显存 / 通信 / HBM）→优化方案 + 参考来源；
- 单轮报表：Markdown 表格，基线 vs 优化后数据、优化点、性能收益；
- 最终报告：全量数据汇总、各优化收益占比、最优生产配置、落地结论。

## 八、输出规范
- 部署 / 调试优先输出可直接运行的带注释 bash 脚本；
- 问题输出：根因 + 修复命令 + 验证指令；
- 报表文档统一 Markdown 表格；
- 禁止输出 CUDA/NVIDIA/GPU 相关方案，全程基于 Ascend+CANN8.5。

## 九、补充说明
后期可接入 SWIFT、VERL 做同环境对标测试；AutoTuning 只用 MindSpeed 原生能力，不引入第三方调优工具；出现多 CANN 环境冲突强制锁定 8.5、屏蔽 8.1 环境变量。