import logging
from numpy.typing import NDArray
from models.interface import Level, ModelService
from models.model_call.models import call_qwen_finetuned, call_vl, call_tts
from os import path, makedirs
from datetime import datetime
from models.prompt import (
    prompt_judge,
    prompt_descirbe_image,
    prompt_bbox,
    prompt_extract_system,
    prompt_extract_example,
    prompt_translate,
    prompt_context,
)
from models.model_call.models import call_qwen_finetuned
import re
from typing import List, Dict, Union, Tuple, Generator
from PIL import Image
import numpy as np
import base64
import io
import json

from aspect import exception_to_logs

logger = logging.getLogger("gradio")

def _nd_to_base64(img: NDArray):
    # 先将NDArray转成base64编码的字符串，便于输入vl模型
    # 根据数组形状猜测色彩模式
    if len(img.shape) == 2:
        # 单通道图像，可能是灰度图
        mode = "L"
    elif img.shape[2] == 3:
        # 三通道图像，RGB
        mode = "RGB"
    elif img.shape[2] == 4:
        # 四通道图像，RGBA
        mode = "RGBA"
    else:
        raise ValueError("Unsupported array shape for conversion to image.")

    # 将NumPy数组转换为PIL图像对象
    pil_image = Image.fromarray(img.astype(np.uint8), mode=mode)

    # 将PIL图像保存到内存中的字节流
    byte_io = io.BytesIO()
    pil_image.save(byte_io, format="PNG")  # 使用PNG格式保持透明度信息（对于RGBA图像）
    byte_io.seek(0)  # 返回到字节流的开头

    # 将字节流编码为Base64字符串
    base64_encoded_data = base64.b64encode(byte_io.getvalue()).decode("utf-8")
    # 加上MIME前缀
    return "data:image;base64," + base64_encoded_data


class ModelServiceDefaultImpl(ModelService):
    """
    ModelService 接口的默认实现类

    依赖组件：
    - model_call.models 模块中的大模型调用函数
    - prompt 模块中的提示词模板

    """

    def test_level(self, message_input: str) -> Level:

        prompt = prompt_judge()

        message = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": message_input},
        ]
        example = call_qwen_finetuned(message)

        # 将字符串转换为 Level 枚举类型
        level_mapping = {
            "A1": Level.A1,
            "A2": Level.A2,
            "B1": Level.B1,
            "B2": Level.B2,
            "C1": Level.C1,
            "C2": Level.C2,
        }

        return level_mapping.get(example, Level.A1)  # 默认返回 Level.A1 如果没有匹配到
        
    def get_img_info(self, img: str|NDArray, level: Level) -> Generator[str | dict, None, None]:
        """
        Desc:
            Return the information of the image, including a description of the image and the words within it along with corresponding bounding boxes, based on the user's level.
            img: base64 encoded image

        Usecase:
            >>> img
            "https..."
            >>> get_img_info(img,Level.A1)
            {"desc":"Laptop bird dustbin cup coffee hit mobilephone" output:"A bird hit a laptop, spilling coffee from a cup. A mobilephone fell in the dustbin.","words":[("laptop",("123","456"),("126","467")),("cup",("53","534"),("86","486"))...]}

        """
        is_base64 = not isinstance(img, str)
        if is_base64:
            img = _nd_to_base64(img)
        # 输出字典
        output_dict = {}
        yield "(0/3) - 正在解析图片..."

        # Part1: image -> vl -> description
        prompt1 = prompt_descirbe_image()
        messages1 = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "根据图片内容描述场景"}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image" if is_base64 else "image_url",
                        "image" if is_base64 else "image_url": img,
                    },
                    {"type": "text", "text": prompt1},
                ],
            },
        ]
        description_list_str = call_vl(messages1)
        description_list = level.name + "," + description_list_str
        yield "(1/3) - 图片解析成功！正在提取对象..."

        # Part2: description_list -> qwen_finetuned -> sentences and words
        prompt_fix = prompt_extract_system()
        message_fix = (
            [{"role": "system", "content": prompt_fix}]
            + prompt_extract_example()
            + [{"role": "user", "content": description_list}]
        )
        info_dict_str = call_qwen_finetuned(message_fix)  # 中英文对应的字典
        info_dict_str_cleaned = info_dict_str.strip(
            "```json\n"
        ).strip()  # 去除字符串首尾的多余字符
        info_dict = json.loads(info_dict_str_cleaned)

        # 解析中英文单词
        en_list = [item[0] for item in info_dict["words"]]
        cn_list = [item[1] for item in info_dict["words"]]

        # 记录输出数据
        output_dict["desc"] = info_dict["desc_en"]
        output_dict["translation"] = info_dict["desc_cn"]
        yield "(2/3) - 对象提取成功！正在生成检测框..."

        # Part3: words -> vl -> bboxes

        prompt4 = prompt_bbox()
        messages2 = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You should print the bboxes of the image.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image" if is_base64 else "image_url",
                        "image" if is_base64 else "image_url": img,
                    },
                    {"type": "text", "text": prompt4 + "input objects:" + str(en_list)},
                ],
            },
        ]
        bbox_str = call_vl(messages2)
        bbox_str_cleaned = bbox_str.strip("```json\n").strip()  # 去除json和换行符
        bbox_list = eval(
            bbox_str_cleaned
        )  # [['blue shirt',('100','200'),('560','665')]...]
        yield "(3/3) - 检测框生成成功！加载中..."

        # 解析检测框数据
        words_list = []
        for index, item in enumerate(bbox_list):
            text = item[0]
            location = [bbox_list[index][1], bbox_list[index][2]]  # 保留元组格式
            translation = cn_list[index]  # 根据索引获取对应的中文翻译

            word_dict = {"text": text, "location": location, "translation": translation}
            words_list.append(word_dict)

        # 记录输出数据
        output_dict["words"] = words_list

        yield "<END>"
        yield output_dict
        """
            {
            "desc":"A bird hit a laptop, spilling coffee from a cup. A mobilephone fell in the dustbin.",
            "translation":"一只鸟撞到了一台平板， 从杯子中溢出了咖啡, 一部手机掉进了垃圾箱。",
           "words":[
                {"text":"laptop","location":[("123","456"),("126","467")],"translation":"平板"},
               { "text":"bird","location":[("100","200"),("110","210")],"translation":"鸟"},
                {"text":"coffee","location":[("300","400"),("310","410")],"translation":"咖啡"},
                {"text":"cup","location":[("320","420"),("330","430")],"translation":"杯子"},
                {"text":"mobilephone","location":[("500","600"),("510","610")],"translation":"手机"},
                {"text":"dustbin","location":[("520","620"),("530","630")],"translation":"垃圾箱"}                    ]
                }
      """

    def get_conversation(
        self, conversation: list, level: Level, img: str | NDArray
    ) -> Generator[str, None, None]:
        is_base64 = not isinstance(img, str)
        if is_base64:
            img = _nd_to_base64(img)

        # 与vl模型进行对话
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a professional image analysis expert and an english teacher.",
                    }
                ],
            }
        ]

        # 添加图片作为第一条用户消息
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image" if is_base64 else "image_url",
                        "image" if is_base64 else "image_url": f"{img}",
                    }
                ],
            }
        )

        # 处理后续对话
        for msg in conversation:
            formatted_msg = {
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}],
            }
            messages.append(formatted_msg)

        context = call_vl(messages)
        # 根据对应等级进行翻译
        context = str(level) + "," + context
        prompt = prompt_translate()
        message = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": context},
        ]
        generator = call_qwen_finetuned(message, True)
        return generator

    def get_new_context(self, words: List[str], level: Level) -> str:
        prompt = prompt_context()
        Add = str(level) + "," + str(words)
        message = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": Add},
        ]
        context = call_qwen_finetuned(message)
        return context

    def get_audio(self, text: str) -> bytes:
        wav = call_tts(text)
        # audio_folder = path.join(models.CACHE_PATH, "generated_audio")

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式：20250122_123456
        # audio_file = path.join(audio_folder, f"{timestamp}.wav")

        # # 创建一个特殊文件夹用于存放音频文件
        # makedirs(audio_folder, exist_ok=True)  # 如果文件夹不存在，则创建

        # # 将生成的音频数据写入文件
        # with open(audio_file, "wb") as f:
        #     f.write(wav)  # 以二进制形式写入音频文件

        # 返回音频文件的本地路径
        # return path.abspath(audio_file)
        return wav
