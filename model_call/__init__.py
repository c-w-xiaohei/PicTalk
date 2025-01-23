from enum import Enum
from typing import Tuple,Generator
from numpy.typing import NDArray
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from IPython.display import Audio, display
from datetime import datetime
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import os

class Level(Enum):
    # CEFR levels
    A1 = 1
    A2 = 2
    B1 = 3
    B2 = 4
    C1 = 5
    C2 = 6

# 初始化微调的大模型，因参数和system提示均为固定，故不写成函数
# 指定模型路径
model_name = "/mnt/workspace/PicTalk/model_call/models/checkpoint-90-gptq-int2" #修改为模型所在checkpoint文件夹的路径
# 检查是否有可用的GPU，如果有的话，使用第一个GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map={"": device})
tokenizer = AutoTokenizer.from_pretrained(model_name)
# system对模型的预设
system = "你是一个英语教学专家。你需要根据CERF英语等级，如“A1,A2”，或者中国中学英语等级，如'junior,senior'，根据给定词汇生成等级对应的英语片段。如：input:'A2,laptop bird dustbin cup coffee hit mobilephone' output:'A bird hit a laptop, spilling coffee from a cup. A mobilephone fell in the dustbin.'"
print("微调模型加载完成，模型名为： " + model_name)


def test_level(message_input: str) -> Level:
    """
    Desc:
        Return the level of the user, according to the composition wrote from the user.
        
    Usecase:
        >>> test_level("We have so many good things to eat for dinner.")
        Level.A1
    """
    torch.cuda.empty_cache()

    # 将输入文本封装为对话格式
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": message_input}
    ]
    
    # 使用模型生成回复
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    
    # 提取生成的文本
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 根据生成的文本判断等级（这里只是一个简单的示例逻辑，可以根据需要调整）
    if "A1" in response:
        return Level.A1
    elif "A2" in response:
        return Level.A2
    elif "B1" in response:
        return Level.B1
    elif "B2" in response:
        return Level.B2
    elif "C1" in response:
        return Level.C1
    elif "C2" in response:
        return Level.C2
    else:
        return Level.A1  # 默认返回 A1 级别

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
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": input_text}
    ]
    
    # 使用模型生成上下文
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    
    # 提取生成的文本
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response


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


# 主函数用于测试，如需测试则把主函数取消注解，重跑init.py
'''
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
'''

