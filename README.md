# 1.概述
  &emsp;&emsp;基于ollama本地模型，把[DSPy](https://dspy.ai/)中的示例代码整合到一个项目中，方便学习和使用。

# 2.安装
    1. 安装ollama
    2. 安装DSPy

# 3.demo运行
## 3.1 ollama服务 
确保ollama服务已启动及依赖的本地模型文件存在。
## 3.2 创建python虚拟环境及安装python依赖
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## 3.3 运行指定的demo
在项目根目录下运行
```
# python -m demos.[章节].[示例]
# 运行flight_agent示例
python -m demos.ai_app.flight_agent
# 运行history示例
python -m demos.ai_app.history

# 运行FT_agents示例
python -m demos.opti_app.FT_agents
```
# 3.代码说明
## 3.1 [tutorials](https://dspy.ai/tutorials/)
&emsp;&emsp;按照tutorials中不同章节的示例代码，到demos目录下对应的子目录中，每一个子目录下有一个readme.md文件，说明了每一个实例及运行测试情况。
|章节|代码目录|
|---|---|
|Build AI Programs with DSPy|ai_app|
|Optimize AI Programs with DSPy|opti_app|
|Reflective Prompt Evolution with dspy.GEPA|reflective_opti|
|Experimental RL Optimization for DSPy|rl_opti|
|Tools, Development, and Deployment|tools_dev_deploy|
|Real-World Examples|real_word|

# 4.典型错误
## 4.1 litellm使用post请求的url，在服务端显示为使用get请求，导致api请求失败
**原因**：
`base_url`值的末尾有’/‘，即base_url="http://localhost:11434/"，传入到litellm中时，url变为`http://localhost:11434//api/chat`
```
# litellm中发起请求的代码
# litellm\llms\custom_httpx\llm_http_handler.py
# async_httpx_client: AsyncHTTPHandler
# _make_common_async_call
...
response = await async_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    data=(
                        signed_json_body
                        if signed_json_body is not None
                        else json.dumps(data)
                    ),
                    timeout=timeout,
                    stream=stream,
                    logging_obj=logging_obj,
                )
...
```
post请求在底层实现中变为实际的get请求，导致api请求失败。
```
# ollama服务端日志
[GIN] 2026/03/26 - 16:35:18 | 405 |            0s |       127.0.0.1 | GET      "/api/chat"
```
**解决方案**：
确保base_url的末尾没有’/‘。即`base_url="http://localhost:11434"`


