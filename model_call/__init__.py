from enum import Enum
from typing import Tuple,Generator
from numpy.typing import NDArray
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from IPython.display import Audio, display
from datetime import datetime
import os

class Level(Enum):
    # CEFR levels
    A1 = 1
    A2 = 2
    B1 = 3
    B2 = 4
    C1 = 5
    C2 = 6
def test_level(message_input:str) -> Level:
    """
        Desc:
            Return the level of the user, according to the composition wrote from the user.
            
        Usecase:
            >>> test_level("We have somany good things to eat for dinner.")
            Level.A1
            
    """
    
    example:Level = Level.A1
    
    return example



def get_img_info(img:NDArray[any],level:Level) -> dict:
    """ 
        Desc:
            Return the information of the image, including a description of the image and the words within it along with corresponding bounding boxes, based on the user's level.
            
        Usecase:
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

def get_new_context(words:list[str],level:Level) -> str:
    """
        Desc:
            Generate a new context description using the given words, based on the user's level.
            
        Usecase:
            >>> get_new_context(["laptop","cup","mobilephone"],Level.A1)
            "There is a laptop, a cup, and a mobile phone."
            
    """
    
    example:str = "There is a laptop, a cup, and a mobile phone."
    return example

# def get_audio(text:str)->str:
#     """
#         Desc:
#             Generate an audio url from the given text.
#         Usecase:
#             >>> get_audio("Hello, how are you?")
#             "https://example.com/audio.mp3"
            
#     """
#     example:str = "https://example.com/audio.mp3"
#     return example


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
    # 指定要使用的模型 ID，这里是一个中文语音合成模型
    model_id = 'iic/speech_sambert-hifigan_tts_zh-cn_16k'

    # 创建一个文本到语音的处理管道，指定任务类型为文本到语音，并使用指定的模型
    sambert_hifigan_tts = pipeline(task=Tasks.text_to_speech, model=model_id)

    # 使用管道处理输入文本，指定使用的语音风格为 'zhitian_emo'
    output = sambert_hifigan_tts(input=text, voice='zhitian_emo')

    # 从输出中提取生成的 WAV 音频数据
    wav = output[OutputKeys.OUTPUT_WAV]

    # 创建一个特殊文件夹用于存放音频文件
    audio_folder = "generated_audio"
    os.makedirs(audio_folder, exist_ok=True)  # 如果文件夹不存在，则创建

    # 使用时间戳生成唯一的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式：20250122_123456
    audio_file = os.path.join(audio_folder, f"{timestamp}.wav")

    # 将生成的音频数据写入文件
    with open(audio_file, 'wb') as f:
        f.write(wav)  # 以二进制形式写入音频文件

    # 返回音频文件的本地路径
    return os.path.abspath(audio_file)