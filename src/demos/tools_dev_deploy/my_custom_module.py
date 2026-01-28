
import dspy

class custom_signature(dspy.Signature):
    """
    自定义签名示例
    """
    question = dspy.InputField(desc="用户的问题")
    answer = dspy.OutputField(desc="模型的回答")