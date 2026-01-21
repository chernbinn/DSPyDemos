# 验证情况

| 文件名 | 简介 | 验证情况 |
| --- | --- | --- |
|**Basic APP**|
|simple_chat.py|1.本地模型配置验证<br>2.简单的dspy moudule调用|测试通过|
|hisitory.py|dspy对历史消息的使用方式；同时，demo展示了不同方式对本地ollama模型的调用|测试完成|
|flight_agent|基于dspy构建的订票智能体||
|custom_dspy_module.py|自定义dspy module示例：在wiki页面搜索内容，然后模型根据搜索结果回答问题|依赖wiki页面，<span style="color:red">国内环境无法访问，无法验证</span>|
|**RAG**|
|local_rag.py|RAG应用的多个简单demos|1.可以运行</br>2.存在模型适配问题，都暂<span style="color:red">未测试通过</span>|
|rag_agent.py|1.根据一个声明，在wiki中找到一致或者反驳的页面标题<br>2.教师-学生模型，教师生成数据后训练学生<br>3.多跳搜索|存在wiki页面访问，<span style="color:red">国内环境无法访问，无法验证</span>|
|rag_agent1.py|1.教师-学生模型，教师生成数据后训练学生</br>2.多个子模型间执行多跳搜索|动态加载huggingface数据，huggingface页面访问难，<span style="color:red">需要重构下载数据到本地</span>|
|**专业用例**|
|entity_extraction.py|提取字段中的实体名称|1.代码逻辑完备<br>2.测试数据需要从huggingface下载，<span style="color:red">暂无法验证</span>|
|classify_events.py|基于事件描述分类的小应用demo|代码逻辑完备，缺乏可测试数据，<span style="color:red">无法验证</span>|
|**高级推理**|
|pot_demos.py|ProgramOfThought demos|部分可验证，易理解可不完全验证|
|**多模态应用**|
|image_generate.py|生成图片|1.代码逻辑完备<br>2.需要替换生成图片的模型，本地暂无ollama模型可用，需要下载有处理图片能力的新模型，<span style="color:red">暂无法验证</span>|
|audio.py|1.基于音频信息回答问题<br>2.基于文字生成指定情绪的音频|1.代码逻辑完备<br>2.需要安装特定的pthon库<br>3.需要替换生成语音的模型，本地暂无ollama模型可用，需要下载有处理语音能力的新模型，<span style="color:red">暂无法验证</span>|


