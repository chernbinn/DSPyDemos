# DSPy优化模型效果

# 验证情况
| 文件名 | 简介 | 验证情况 |
| --- | --- | --- |
|数学及推理任务|
|math_reasoning.py|1.使用COT处理数学问题 <br>2.验证模型在MATH数据集上的性能<br>3.通过教师-学生模型优化提示策略|测试过程可以完成。<br><span style="color:red">结果不符合预期，测试过程有待分析</span>|
|模型优化|
|classify_ft.py|1.通过教师模型蒸馏微调分类模型<br>2.分别使用未标记数据和已标记数据进行微调<br>3.对比不同微调策略的效果|<span style="color:red">数据无法下载，无法测试</span>|      
|高级工具集成|
|advanced_tooluse.py|探索如何优化使用外部工具和API的AI程序|<span style="color:red">数据无法下载，无法测试</span>|
|FT_agents.py|基于蒸馏微调优化基于代理的复杂系统，学习如何在像游戏这样的交互环境中提高代理性能|<span style="color:red">依赖环境无法在windows上搭建，无法测试</span>|
