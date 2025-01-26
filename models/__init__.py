"""
PicTalk 核心模型服务模块

1. Level 枚举 - CEFR语言能力等级标准（A1-C2）
2. ModelService 抽象接口 - 定义模型服务核心功能
3. 默认实现实例 - ModelServiceDefaultImpl 的全局实例

"""


from .impl import ModelServiceDefaultImpl
from .interface import ModelService, Level

# 实例化具体实现并对外暴露
service: ModelService = ModelServiceDefaultImpl()

# 对外暴露Level枚举类
__all__ = ['service', 'Level']


# 主函数用于测试，如需测试则把主函数取消注解，重跑init.py
"""
if __name__ == "__main__":
    # 测试 test_level 函数
    user_input = "We have so many good things to eat for dinner."
    level = service.test_level(user_input)

    # 测试 get_new_context 函数
    words = ["laptop", "cup", "mobilephone"]
    context = service.get_new_context(words, Level.A1)
    
    # 测试 get_audio 函数
    test_text = "今天天气很晴朗"
    audio_path = service.get_audio(test_text)
    
    # 三个函数的输出
    print(f"Detected Level: {level}") # 应该输出输入句子的难度等级
    print(f"Generated Context: {context}") # 应该输出根据难度等级和输入词汇造出的句子
    print(f"Generated audio file path: {audio_path}") # 应该输出生成音频wav文件的路径
"""


