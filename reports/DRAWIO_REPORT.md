# DrawIO 图示生成报告

## 图示清单

| 文件 | 类型 | 来源依据 | 用途 | 状态 |
|------|------|---------|------|------|
| fig_roadmap.drawio | 技术路线图 | ANALYSIS_MODELING_REPORT.md | 论文引言/总体思路 | 已生成源文件, PDF待导出 |
| fig_flow_q1.drawio | 求解流程图 | 问题一模型 | 问题一建模章节 | 已生成源文件 |

## 未生成图示及原因

- fig_pipeline（数据处理流程图）：数据处理较简单（折射率插值 + 光谱数据加载），在正文中用文字描述即可
- fig_flow_q3/q4（问题三/四流程图）：与问题一/二的TMM方法一致，可复用描述

## 导出与自检记录

- DrawIO CLI未安装在当前环境，无法自动导出PDF
- 建议用户使用 https://app.diagrams.net/ 在线打开 .drawio 文件，手动导出为PDF
- 导出命令（如已安装drawio）: `drawio --export --format pdf --crop --output figures/fig_roadmap.pdf figures/fig_roadmap.drawio`

## 给论文阶段的嵌入建议

- fig_roadmap.pdf → 放入引言/问题分析章节，展示整体解题思路
- fig_flow_q1.pdf → 放入问题一建模章节，展示TMM求解流程
