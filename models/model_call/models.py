import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer,AutoProcessor,snapshot_download,Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from transformers import TextIteratorStreamer
from threading import Thread

from enum import Enum
from os import path,environ
import json 
from typing import Optional,Generator

# models 目录绝对路径
MODELS_PATH = path.dirname(__file__)

# checkpoint 文件夹的绝对路径
CHEKPOINT_PATH = path.join(MODELS_PATH,"checkpoint-90-gptq-int2")

# 模型缓存地址
CACHE_PATH = path.join(MODELS_PATH,".cache")
environ['MODELSCOPE_CACHE'] = CACHE_PATH

class Model(Enum):
    INSTURCT = "qwen/Qwen2.5-7B-Instruct"
    VL = "qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"
    TTS = "iic/speech_sambert-hifigan_tts_zh-cn_16k"

def _get_model(requested_model:Model) -> str :
    """ 根据模型枚举给出模型的实际存储地址 """
    try:
        model_path = snapshot_download(requested_model.value)
    except Exception as e:
        raise ValueError(f"模型加载时出现异常：{e}")
    return model_path
    
def call_qwen_finetuned(messages:list,stream:bool = False) -> str | Generator[str,None,None]:
    """
    微调大模型调用函数
    stream:是否使用流式输出
    """

    torch.cuda.empty_cache()

    # 模型绝对路径
    base_model_path = _get_model(Model.INSTURCT)
    model_path = CHEKPOINT_PATH

    # 更改基础模型路径的配置
    config_path = path.join(model_path,"adapter_config.json")
    with open(config_path, 'r') as file:
        config = json.load(file)
    config["base_model_name_or_path"] = base_model_path
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)
        
    # 检查是否有可用的GPU，如果有的话，使用第一个GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map={"": device})
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("微调模型加载完成，模型名为： " + model_path)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        streamer=stream
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    if not stream:
        # 直接生成响应
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        
        # 提取生成的文本
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response =  tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response
    else:
        # 输出流式响应
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # 在单独的线程中调用.generate()
        generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=100)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer

def call_vl(messages:dict)->str:
    """
    messages 示例格式：
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image":  "data:image;base64,/9j/..."
                },
                {"type": "text", "text": '''使用最简洁的话请描述这张图片的场景。'''},
            ],
        }
    ]
    """
    model_dir = _get_model(Model.VL)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
      model_dir, torch_dtype="auto", device_map="auto"
    )

    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)


    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text

def call_tts(text:str)->bytes: 

    # 创建一个文本到语音的处理管道，指定任务类型为文本到语音，并使用指定的模型
    sambert_hifigan_tts = pipeline(task=Tasks.text_to_speech, model=Model.TTS.value)

    # 使用管道处理输入文本，指定使用的语音风格为 'zhitian_emo'
    output = sambert_hifigan_tts(input=text, voice='zhitian_emo')

    # 从输出中提取生成的 WAV 音频数据
    wav = output[OutputKeys.OUTPUT_WAV]

    return wav
