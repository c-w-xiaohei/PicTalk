from enum import Enum
from typing import Tuple,Generator
from numpy.typing import NDArray


from IPython.display import Audio, display

from model_call import models
from os import path,makedirs
from datetime import datetime

from model_call.models import call_qwen_finetuned
from prompt import prompt_first
class Level(Enum):
    # CEFR levels
    A1 = 1
    A2 = 2
    B1 = 3
    B2 = 4
    C1 = 5
    C2 = 6


def test_level(message_input: str) -> Level:
    """
    Desc:
        Return the level of the user, according to the composition wrote from the user.
        
    Usecase:
        >>> test_level("We have so many good things to eat for dinner.")
        Level.A1
    """
    prompt=prompt_first()
   
    message = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": message_input}
    ]
    example = call_qwen_finetuned(message)
    
    # 将字符串转换为 Level 枚举类型
    level_mapping = {
        "A1": Level.A1,
        "A2": Level.A2,
        "B1": Level.B1,
        "B2": Level.B2,
        "C1": Level.C1,
        "C2": Level.C2
    }
    
    return level_mapping.get(example,Level.A1)  # 默认返回 Level.A1 如果没有匹配到

def get_img_info(img:str,level:Level) -> dict:
    """ 
        Desc:
            Return the information of the image, including a description of the image and the words within it along with corresponding bounding boxes, based on the user's level.
            img: base64 encoded image
            
        Usecase:
            >>> img
            "data:image;base64,/9j/..."
            >>> get_img_info(img,Level.A1)
            {"desc":"Laptop bird dustbin cup coffee hit mobilephone" output:"A bird hit a laptop, spilling coffee from a cup. A mobilephone fell in the dustbin.","words":[("laptop",("123","456"),("126","467")),("cup",("53","534"),("86","486"))...]}

    """
    
    example:dict = {
    "desc":"The description of the image",
    "words":[
        ("word1",("x1","y1"),("x2","y2")),
        ("word2",("x1","y1"),("x2","y2")),
        ("word3",("x1","y1"),("x2","y2"))
        ]
    }
    
    return example

def get_conversation(conversation:list,level:Level,img:NDArray[any]) -> Generator[str,None,None]:
    """
        Desc:
            Return the generator of conversation, including the user's question and the assistant's answer, based on the user's level.
            
        Usecase:
            >>> chat = [
                {"role": "user", "content": "Hello"}, 
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "What do you see in the image?"}
            ]
            >>> get_chat(chat,Level.A1,img)
            (a generator of response string)
    """
    
    exmample_arr = [
        "Hello.","What","do","you","see?"
    ]    
    
    for s in exmample_arr:
        yield s


def get_new_context(words: list[str], level: Level) -> str:
    """
    Desc:
        Generate a new context description using the given words, based on the user's level.
        
    Usecase:
        >>> get_new_context(["laptop", "cup", "mobilephone"], Level.A1)
        "There is a laptop, a cup, and a mobile phone."
    """
    # 将输入词汇和等级封装为对话格式
    input_text = f"{level.value},{','.join(words)}"



def get_audio(text: str) -> str:
    """
    Desc:
        Generate an audio file from the given text and return the local file path.
        The audio file will be saved in a specific folder with a unique name based on the current timestamp.
    Usecase:
        >>> audio_path = get_audio("今天天气很晴朗")
        >>> print(audio_path)
        "/path/to/audio/20250122_123456.wav"
    """

    wav = models.call_tts(text)
    audio_folder = path.join(models.CACHE_PATH,"generated_audio")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式：20250122_123456
    audio_file = path.join(audio_folder, f"{timestamp}.wav")
    
    # 创建一个特殊文件夹用于存放音频文件
    makedirs(audio_folder, exist_ok=True)  # 如果文件夹不存在，则创建

    # 将生成的音频数据写入文件
    with open(audio_file, 'wb') as f:
        f.write(wav)  # 以二进制形式写入音频文件

    # 返回音频文件的本地路径
    return path.abspath(audio_file)




# 主函数用于测试，如需测试则把主函数取消注解，重跑init.py
"""
if __name__ == "__main__":
    # 测试 test_level 函数
    user_input = "We have so many good things to eat for dinner."
    level = test_level(user_input)

    # 测试 get_new_context 函数
    words = ["laptop", "cup", "mobilephone"]
    context = get_new_context(words, Level.A1)
    
    # 测试 get_audio 函数
    test_text = "今天天气很晴朗"
    audio_path = get_audio(test_text)
    
    # 三个函数的输出
    print(f"Detected Level: {level}") # 应该输出输入句子的难度等级
    print(f"Generated Context: {context}") # 应该输出根据难度等级和输入词汇造出的句子
    print(f"Generated audio file path: {audio_path}") # 应该输出生成音频wav文件的路径
"""
