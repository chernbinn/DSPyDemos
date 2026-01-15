import dspy
from litellm import completion, acompletion


class OllamaLMLiteLLM(dspy.LM):
    """
    使用LiteLLM实现的Ollama模型适配器
    LiteLLM提供统一的API接口，自动处理OpenAI格式转换
    """
    
    def __init__(self, model="llama3", base_url="http://192.168.3.17:11111", max_tokens=500, temperature=0.7, **kwargs):
        """
        初始化Ollama模型适配器
        :param model: 模型名称（如llama3.2:3b）
        :param base_url: Ollama服务器地址
        :param max_tokens: 最大生成 tokens 数
        :param temperature: 温度参数
        :param kwargs: 其他参数
        """
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.kwargs = kwargs
        
        # 设置litellm的配置
        import litellm
        litellm.set_verbose=True
        litellm.api_base = self.base_url
        
        # 调用BaseLM初始化
        super().__init__(model=model, model_type="chat")
        
        # 确保self.model包含ollama/前缀（覆盖BaseLM设置的值）
        self.model = f"ollama/{model}"
    
    def forward(self, prompt=None, messages=None, **kwargs):
        """
        实现BaseLM要求的forward方法
        使用LiteLLM调用Ollama API，自动处理格式转换
        """
        # 合并参数
        merged_kwargs = {**self.kwargs, **kwargs}
        
        try:
            # 从merged_kwargs中获取参数，避免重复传递
            max_tokens = merged_kwargs.pop("max_tokens", self.max_tokens)
            temperature = merged_kwargs.pop("temperature", self.temperature)
            
            # 使用litellm调用Ollama API
            response = completion(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **merged_kwargs  # 此时merged_kwargs中已经没有max_tokens和temperature
            )
            
            return response
        except Exception as e:
            print(f"LiteLLM调用错误: {e}")
            raise
    
    async def aforward(self, prompt=None, messages=None, **kwargs):
        """
        异步版本的forward方法
        """
        merged_kwargs = {**self.kwargs, **kwargs}
        
        try:
            # 从merged_kwargs中获取参数，避免重复传递
            max_tokens = merged_kwargs.pop("max_tokens", self.max_tokens)
            temperature = merged_kwargs.pop("temperature", self.temperature)
            
            response = await acompletion(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **merged_kwargs  # 此时merged_kwargs中已经没有max_tokens和temperature
            )
            
            return response
        except Exception as e:
            print(f"异步LiteLLM调用错误: {e}")
            raise
