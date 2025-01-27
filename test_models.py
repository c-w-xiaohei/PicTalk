from models import ModelServiceDefaultImpl,Level


if __name__ == "__main__":
    # 实例化对象
    test = ModelServiceDefaultImpl()
    # 测试 test_level 函数
    user_input = "We have so many good things to eat for dinner."
    level = test.test_level(user_input)

    # 测试 get_new_context 函数
    words = ["laptop", "cup", "mobilephone"]
    context = test.get_new_context(words, Level.A1)
    
    # 测试 get_audio 函数
    test_text = "今天天气很晴朗"
    audio_path = test.get_audio(test_text)
    
    # 三个函数的输出
    print(f"Detected Level: {level}") # 应该输出输入句子的难度等级
    print(f"Generated Context: {context}") # 应该输出根据难度等级和输入词汇造出的句子
    print(f"Generated audio file path: {audio_path}") # 应该输出生成音频wav文件的路径
