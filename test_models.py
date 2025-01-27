from models import ModelServiceDefaultImpl,Level

"""
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
"""
# test_models.py
import sys
import os

def read_input_from_file(file_name):
    """
    从与脚本同目录下的指定文件读取输入，并返回读取的内容作为一个字符串。
    
    :param file_name: 文件名，程序将从此文件读取输入
    :return: 包含文件所有内容的单个字符串
    """
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建文件的完整路径
    file_path = os.path.join(current_dir, file_name)
    
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"找不到指定的文件: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()  # 读取整个文件内容到一个字符串
        return content
    except IOError as e:
        print(f"读取文件时发生错误: {e}")
        return ""

if __name__ == "__main__":
    # 实例化对象
    test = ModelServiceDefaultImpl()
    # 测试 get_img_info 函数
    input_file_name = 'Input.txt'
    a = ""  # 初始化变量a，用于接收文件内容作为字符串
    
    try:
        a = read_input_from_file(input_file_name)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)  # 如果文件未找到，退出程序
    
    level = Level.A1
    context = test.get_img_info(a, level)  # 通过实例调用 get_img_info 方法并传递 img 和 level 参数
    context = str(context)
    print(context)



