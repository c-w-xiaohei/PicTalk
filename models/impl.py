from typing import Generator, List, Dict
from numpy.typing import NDArray
from models.interface import Level, ModelService
from models.model_call.models import call_qwen_finetuned,call_vl
from os import path, makedirs
from datetime import datetime
from models.prompt import prompt_first,  prompt_second_first,  prompt_second_second,  prompt_second_third,  prompt_second_forth, prompt_second_fix
from models.model_call.models import call_qwen_finetuned
import re
from typing import List, Dict, Union, Tuple, Generator
#内置函数，将字符串转换为列表，字符串样例：[[objectA,(x1,y1),(x2,y2)],[objectB,(x1,y1),(x2,y2)],...untill no objects left]
def _parse_detection_boxes(detection_boxes_str: str) -> List[Dict[str, Union[str, List[Tuple[int, int]]]]]:
  try:

        # 去除外层方括号
        detection_boxes_str = detection_boxes_str.strip()[1:-1]
        
        # 分割每个对象的条目
        entries = detection_boxes_str.split('], [')
        
        # 初始化结果列表
        result_list = []
        
        for entry in entries:
            # 去掉可能存在的前后空格和多余的方括号
            entry = entry.strip().strip('[]')
            
            # 分割对象名称和坐标
            parts = entry.split(', ')
            
            if len(parts) != 3:
                continue
            
            object_name = parts[0].strip("'")
            
            # 解析坐标
            coord1 = tuple(map(int, parts[1].strip('()').split(',')))
            coord2 = tuple(map(int, parts[2].strip('()').split(',')))
            
            # 构建字典项
            result_list.append({"word": object_name, "coordinates": [coord1, coord2]})
        
        return result_list
    
  except Exception as e:
        print(f"解析检测框字符串时出错: {e}")
        return []
  

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
    def get_img_info(self,img:str,level:Level) -> dict:
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
     #Part 1
        prompt1 = prompt_second_first()
        messages1 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img
                },
                {"type": "text", "text": prompt1},
            ],
        }
    ]
        List = call_vl(messages1)
#Part 2
        prompt_fix = prompt_second_fix()
        message_fix =  [
        {"role": "system", "content": prompt_fix},
        {"role": "user", "content": List}
    ]
        List_words = call_qwen_finetuned(message_fix)
        prompt2 = prompt_second_second
        Add = str(level) + "," + List_words
        message2 = [
        {"role": "system", "content": prompt2},
        {"role": "user", "content": Add}
    ]
        sentence = call_qwen_finetuned(message2,False)
        eng_pattern = r'[A-Za-z\s\.\,\-\']+'  # 匹配英文字符及常见标点
        zh_pattern = r'[\u4e00-\u9fff]+'      # 匹配中文字符
    
        english_parts = re.findall(eng_pattern, sentence)
        chinese_parts = re.findall(zh_pattern, sentence)
    
    # 将找到的部分合并为完整的句子
        english_sentence = ' '.join(english_parts).strip()
        chinese_sentence = ''.join(chinese_parts)
        prompt3 = prompt_second_third()
        message2 = [
        {"role": "system", "content": prompt3},
        {"role": "user", "content": english_sentence}
    ]
#Paet3
        prompt4 = prompt_second_forth()
        messages2 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img
                },
                {"type": "text", "text": prompt4},
            ],
        }
    ]
        Part3 = call_vl(messsages2)
#Part4
        List_list = _parse_detection_boxes(Part3)
        output_dict = {
        "desc": english_sentence ,
        "translation": chinese_sentence,
        "words": List_list
    }
        return output_dict
   
    """  
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
    """
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