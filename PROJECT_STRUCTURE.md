# CAM++ 项目主索引（唯一结构入口）

本文档是仓库结构与执行流程的唯一主索引。
原先重复的结构说明文档已收敛为跳转页，后续以本文件为准。

## 1. 仓库分层

```
camplusplus/
├── egs/3dspeaker/sv-cam++/     # Recipe 层：实验编排与数据准备入口
├── speakerlab/                 # 框架层：训练/推理/模型/数据处理核心实现
├── my_methods/                 # 自定义方法：CA-AFC 前端与配套脚本
├── note/                       # 实验记录、命令手册、结果模板
└── pretrained/                 # 预训练模型与模型说明
```

## 2. 核心调用链（基线）

```
egs/3dspeaker/sv-cam++/run.sh
  -> local/prepare_data.sh
  -> local/prepare_data_csv.py
  -> speakerlab/bin/train.py
  -> speakerlab/bin/extract.py
  -> speakerlab/bin/compute_score_metrics.py
```

## 3. 关键目录职责

- `egs/3dspeaker/sv-cam++/conf/`: 训练配置（如 `cam++.yaml`）
- `egs/3dspeaker/sv-cam++/local/`: 数据下载、数据准备、codec 评测脚本
- `speakerlab/bin/`: 训练、提取、评分、推理入口
- `speakerlab/models/campplus/`: CAM++ 主模型与分类头
- `my_methods/models/`: CA-AFC 等自定义前端模型
- `my_methods/scripts/`: CA-AFC 训练与评测入口

## 4. 文档导航

- Recipe 官方说明：`egs/3dspeaker/sv-cam++/README.md`
- 容器 Linux 命令总手册（统一入口）：`note/container_linux_runbook.md`
- 服务器部署与流程说明：`my_methods/note/server_deployment_workflow.md`
- 实验记录模板：`note/Experiment_Table_Template.md`

## 5. 维护约定

1. 结构相关内容只在本文件更新。
2. 命令相关内容统一维护在 `note/container_linux_runbook.md`。
3. 其他文档若涉及结构或命令，仅保留简述并链接到主文档，避免重复维护。
