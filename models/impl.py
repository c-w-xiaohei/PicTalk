from typing import Generator, List, Dict
from numpy.typing import NDArray
from . import Level, ModelService
from model_call import models
from os import path, makedirs
from datetime import datetime
from prompt import prompt_first
from model_call.models import call_qwen_finetuned

class ModelServiceDefaultImpl(ModelService):
    """
    ModelService 接口的默认实现类

    依赖组件：
    - model_call.models 模块中的大模型调用函数
    - prompt 模块中的提示词模板
    
    """
    def test_level(self, message_input: str) -> Level:

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

    def get_img_info(self, img: str, level: Level) -> Dict:
        return {
            "desc": "The description of the image",
            "translation": "图片的描述",
            "words": [
                ("word1", ("x1", "y1"), ("x2", "y2")),
                ("word2", ("x1", "y1"), ("x2", "y2")),
                ("word3", ("x1", "y1"), ("x2", "y2"))
            ],
            "words_translation": ["单词1", "单词2", "单词3"]
        }

    def get_conversation(self, conversation: list, level: Level, img: NDArray) -> Generator[str, None, None]:
        exmample_arr = ["Hello.", "What", "do", "you", "see?"]
        for s in exmample_arr:
            yield s

    def get_new_context(self, words: List[str], level: Level) -> str:
        input_text = f"{level.value},{','.join(words)}"
        return input_text

    def get_audio(self, text: str) -> str:
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