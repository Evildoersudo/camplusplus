# sv-cam++ 目录说明（精简版）

本文件保留最小说明，详细结构已统一到主索引。

## 目录定位

`egs/3dspeaker/sv-cam++` 是 CAM++ baseline 的实验编排目录，负责串联：

1. 数据准备
2. 训练 CSV 生成
3. 模型训练
4. embedding 提取
5. 指标评测

## 主流程

```bash
bash run.sh --stage 1 --stop_stage 5
```

## 常用入口

- 主入口：`run.sh`
- 配置：`conf/cam++.yaml`
- 数据准备：`local/prepare_data.sh`
- 训练入口（框架层）：`../../../speakerlab/bin/train.py`

## 权威文档

- 结构主索引：`../../../PROJECT_STRUCTURE.md`
- 容器 Linux 命令手册：`../../../note/container_linux_runbook.md`
