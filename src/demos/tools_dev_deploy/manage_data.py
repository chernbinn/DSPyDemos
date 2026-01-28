'''
本指南演示了如何保存和加载您的DSPy程序。从高层次来看，保存您的DSPy程序有两种方法：

1.仅保存程序的状态，类似于PyTorch中的仅保存权重。
2.保存整个程序，包括架构和状态，这由dspy>=2.6.0支持。
'''

import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

gsm8k = GSM8K()
gsm8k_trainset = gsm8k.train[:10]
dspy_program = dspy.ChainOfThought("question -> answer")

optimizer = dspy.BootstrapFewShot(metric=gsm8k_metric, max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=5)
compiled_dspy_program = optimizer.compile(dspy_program, trainset=gsm8k_trainset)

def save_state_only():
    '''
    仅保存程序的状态，类似于PyTorch中的仅保存权重。
    '''
    print("仅保存程序状态到 program.json")
    compiled_dspy_program.save("./dspy_program/program.json", save_program=False)
    # 加载 .pkl 文件可以执行任意代码，可能存在危险。仅在安全环境下从可信来源加载 pickle 文件。尽可能优先使用 .json 文件。
    # 如果必须使用 pickle 文件，请确保信任来源，并在加载时使用 allow_pickle=True 参数。
    print("仅保存程序状态到 program.pkl")
    compiled_dspy_program.save("./dspy_program/program.pkl", save_program=False)

    print("从 program.json 加载程序状态")
    loaded_dspy_program = dspy.ChainOfThought("question -> answer") # Recreate the same program.
    loaded_dspy_program.load("./dspy_program/program.json")

    print("验证加载json的程序状态是否与原始程序状态相同")
    assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
    for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
        # Loaded demo is a dict, while the original demo is a dspy.Example.
        assert original_demo.toDict() == loaded_demo
    assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)

    print("从 program.pkl 加载程序状态")
    loaded_dspy_program = dspy.ChainOfThought("question -> answer") # Recreate the same program.
    loaded_dspy_program.load("./dspy_program/program.pkl", allow_pickle=True)

    print("验证加载pkl的程序状态是否与原始程序状态相同")
    assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
    for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
        # Loaded demo is a dict, while the original demo is a dspy.Example.
        assert original_demo.toDict() == loaded_demo
    assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)

'''
# 整个程序保存使用 cloudpickle 进行序列化，这和 Pickle 文件有相同的安全风险。仅在安全环境下从可信来源加载程序。
从 dspy>=2.6.0 开始，DSPy 支持保存整个程序，包括架构和状态。该功能由 cloudpickle 提供支持，
它是一个用于序列化和反序列化 Python 对象的库。
要保存整个程序，使用 save 方法并设置 save_program=True ，并指定一个目录路径来保存程序而不是文件名。
我们要求目录路径，因为我们也保存一些元数据，例如程序本身的依赖版本。
要加载保存的程序，直接使用 dspy.load 方法。
通过整个程序保存，你无需重新创建程序，可以直接加载架构以及状态。
'''
def save_entire_program():
    print("保存整个程序到 ./dspy_program/目录")
    compiled_dspy_program.save("./dspy_program/", save_program=True)
    print("从 ./dspy_program/目录 加载整个程序")
    # 加载整个程序
    loaded_dspy_program = dspy.load("./dspy_program/")

    print("验证加载整个程序的状态是否与原始程序状态相同")
    assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
    for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
        # Loaded demo is a dict, while the original demo is a dspy.Example.
        assert original_demo.toDict() == loaded_demo
    assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)

'''
注意：
1.当使用 save_program=True 保存程序时，你可能需要包含程序所依赖的自定义模块。如果程序依赖这些模块，
但在调用 dspy.load 之前没有导入这些模块，这是必要的。
2.您可以通过在调用 save 时将它们传递给 modules_to_serialize 参数来指定哪些自定义模块应该与您的程序
一起序列化。这确保了您的程序所依赖的任何依赖项在序列化期间都包含在内，并且在稍后加载程序时可用。

在底层，这使用 cloudpickle 的 cloudpickle.register_pickle_by_value 函数将模块注册为可按值序列
化的模块。当以这种方式注册模块时，cloudpickle 将按值序列化模块，而不是按引用序列化，确保模块内容与保存的程序一起被保留。
'''
def save_program_with_custom_module():
    print("保存整个程序到 ./dspy_program/目录，包含自定义模块 my_custom_module")
    from demos.tools_dev_deploy import my_custom_module

    compiled_dspy_program = dspy.ChainOfThought(my_custom_module.custom_signature)

    # Save the program with the custom module
    compiled_dspy_program.save(
        "./dspy_program/",
        save_program=True,
        modules_to_serialize=[my_custom_module]
    )
'''
这确保了所需的模块在稍后加载程序时能够被正确序列化并可用。可以传递任意数量的模块给 modules_to_serialize 。
如果你不指定 modules_to_serialize ，就不会为序列化注册任何额外的模块。

注意：
1.截至目前 dspy<3.0.0 ，我们不再保证保存的程序具有向后兼容性。例如，如果你使用 dspy==2.5.35 
保存程序，在加载时请确保使用相同版本的 DSPy 来加载程序，否则程序可能无法按预期运行。很可能在不
同版本的 DSPy 中加载保存的文件不会报错，但性能可能与保存时有所不同。

2.从 dspy>=3.0.0 版本开始，我们将保证在主要版本中保存的程序具有向后兼容性，即 dspy==3.0.0 版本中保存的
程序应该可以在 dspy==3.7.10 版本中加载。
'''