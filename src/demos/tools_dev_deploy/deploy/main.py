
'''
本指南展示了两种将 DSPy 程序部署到生产环境中的潜在方法：使用 FastAPI 进行轻量级部署，
以及使用 MLflow 进行更高级的生产级部署，包括程序版本管理和控制。

MLflow 是一个用于管理机器学习实验和部署模型的平台。它提供了一个统一的界面，
用于跟踪实验、版本控制模型、部署模型以及监控模型性能。此处不考虑。
pip install fastapi uvicorn
'''

import requests

def test_fastapi_dspy():
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"text": "What is the capital of France?"}
    )
    print(response.json())
    assert response.json()["status"] == "success"

