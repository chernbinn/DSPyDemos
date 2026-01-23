
# 基于反思算法优化模型
本节介绍了GEPA，一个用于DSPy的反思提示优化器。GEPA通过利用语言模型对DSPy程序轨迹的反思能力，识别哪些方面做得好，哪些方面不好，以及哪些方面可以改进。基于这种反思，GEPA提出新的提示，构建一个不断演进的提示候选树，随着优化的进行积累改进。由于GEPA可以利用特定领域的文本反馈（而不仅仅是标量指标），因此GEPA通常能在很少的回滚中提出高性能的提示。GEPA在论文《GEPA：反思提示演化可以优于强化学习》中介绍，并且可以通过dspy.GEPA使用，其内部使用gepa-ai/gepa提供的GEPA实现。

# demos验证情况
| 文件名 | 简介 | 验证情况 |
| --- | --- | --- |
|math.py|使用反思算法优化数学问题求解模型|数据加载成功，评估过程出现错误，需要深入分析DSPy代码，可能存在适配需求。<span style="color:red">测试失败</span>|
|info_extract.py|使用反思算法优化信息提取模型|<span style="color:red">数据下载失败，无法测试</span>|
|privacy_delegate.py|基于反思算法优化隐私代理模型||
|ai_control_classify.py|基于反思算法优化AI控制分类模型|<span style="color:red">代码复杂、环境依赖复杂，延后分析验证</span>|