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

import multiprocessing
import logging


"""
- 路径配置

"""

# models 目录绝对路径
MODELS_PATH = path.dirname(__file__)

# checkpoint 文件夹的绝对路径
CHEKPOINT_PATH = path.join(MODELS_PATH,"checkpoint-90-gptq-int2")

# 模型缓存地址
CACHE_PATH = path.join(MODELS_PATH,".cache")
environ['MODELSCOPE_CACHE'] = CACHE_PATH


"""
- 日志配置

"""
logging.basicConfig(filename=path.join(MODELS_PATH,"models.log"), filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

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

    

"""
- 进程配置

"""

PROCESS_CONFIG = {
    Model.INSTURCT: False,
    Model.VL: True,
    Model.TTS: False
}

def _run_in_process(func, queue, *args, **kwargs):
    """ 在子进程中执行函数并捕获结果/异常 """
    try:
        result = func(*args, **kwargs)
        queue.put(result)
    except Exception as e:
        queue.put(e)

def call_qwen_finetuned(messages: list, stream: bool = False) -> str | Generator[str, None, None]:
    logging.info(f'''调用微调模型:
            @messages:{messages}
            @stream{stream}''')
    if PROCESS_CONFIG.get(Model.INSTURCT, False):
        if stream:
            raise ValueError("Stream mode not supported in process mode")
            
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()
        p = ctx.Process(
            target=_run_in_process,
            args=(_call_qwen_finetuned, queue, messages, False)
        )
        p.start()
        p.join()
        
        result = queue.get()
        if isinstance(result, Exception):
            raise result
        return result
    else:
        return _call_qwen_finetuned(messages, stream=stream)

def call_vl(messages: dict) -> str:
    log_msg = str(messages)
    if log_msg.__len__()>300:
        log_msg = log_msg[:300] + "..."
    logging.info(f'''调用视觉模型:
        @messages:{log_msg}''')
    if PROCESS_CONFIG.get(Model.VL, False):
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()
        p = ctx.Process(
            target=_run_in_process,
            args=(_call_vl, queue, messages)
        )
        p.start()
        p.join()
        
        result = queue.get()
        if isinstance(result, Exception):
            raise result
        return result
    else:
        return _call_vl(messages)

def call_tts(text: str) -> bytes:
    logging.info(f'''调用视觉模型:
        @text:{text}''')
    if PROCESS_CONFIG.get(Model.TTS, False):
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()
        p = ctx.Process(
            target=_run_in_process,
            args=(_call_tts, queue, text)
        )
        p.start()
        p.join()
        
        result = queue.get()
        if isinstance(result, Exception):
            raise result
        return result
    else:
        return _call_tts(text)
def _call_qwen_finetuned(messages:list,stream:bool = False) -> str | Generator[str,None,None]:
    """
    微调大模型调用函数
    stream:是否使用流式输出
    """
    
    # 验证messages格式
    if not isinstance(messages, list):
        raise ValueError("参数类型错误:messages参数必须为list类型")
    
    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError("参数类型错误:messages中的每个元素必须为dict类型")
        if "role" not in msg or "content" not in msg: # 每个值需要是字符串
            raise ValueError("参数类型错误:messages中的每个元素必须包含'role'和'content'键")
        if not isinstance(msg["role"], str) or not isinstance(msg["content"], str):
            raise ValueError("参数类型错误:messages中的'role'和'content'值必须为字符串类型")
    
    try:
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
        logging.info("微调大模型加载中...")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map={"": device})
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logging.info("微调大模型加载成功")


        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            streamer=stream
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        if not stream:
            # 直接生成响应
            logging.info("微调大模型开始推理...")

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            
            # 提取生成的文本
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response =  tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            torch.cuda.empty_cache()
            logging.info(f'''微调大模型推理完毕:
                @response{response}''')
            return response
        else:
            # 输出流式响应
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            # 在单独的线程中调用.generate()
            generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=100)
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            torch.cuda.empty_cache()
            
            logging.info(f'''微调大模型推理完毕:
                @response(流式输出模式){streamer}''')
            return streamer
    except torch.cuda.OutOfMemoryError as e:
        raise MemoryError(f"model_call 异常: 微调大模型调调用过程显存不足，请检查是否使用了GPU：\n{e}")
    except Exception as e:
        raise Exception(f"model_call 异常: 微调大模型调用过程中出现异常：\n{e}")

def _call_vl(messages:dict)->str:
    """
    messages 示例格式：
    messages = [
  {
    "role": "system",
    "content": [
      {
        "type": "text",
        "text": "You are a helpful assistant."
      }
    ]
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image",
        "image": "data:image;base64,/9j/..."
      },
      {
        "type": "text",
        "text": "'使用最简洁的话请描述这张图片的场景。'"
      }
    ]
  },
  {
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "这张图片展示了一座建筑前的雕像，雕像位于一个基座上，背景是一座现代建筑。"
      }
    ]
  },
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "请描述图片的对象"
      }
    ]
  }...
]
    """
    # 验证messages格式
    if not isinstance(messages, list):
        raise ValueError("参数类型错误:messages参数必须为list类型")
    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError("参数类型错误:messages中的每个元素必须为dict类型")
        if "role" not in msg or "content" not in msg:
            raise ValueError("参数类型错误:messages中的每个元素必须包含'role'和'content'键")
        if not isinstance(msg["content"], list):
            raise ValueError("参数类型错误:messages中的'content'值必须为list类型")
        for item in msg["content"]:
            if not isinstance(item, dict):
                raise ValueError("参数类型错误:messages中的'content'列表中的元素必须为dict类型")
            if "type" not in item:
                 raise ValueError("参数类型错误:messages中的'content'列表中的元素必须包含'type'键")
            if item["type"] not in ["image", "text"]:
                raise ValueError("参数类型错误:messages中的'content'列表中的'type'值必须为'image'或'text'")
            if item["type"] == "image":
                if "image" not in item:
                    raise ValueError("参数类型错误:当'type'为'image'时，必须包含'image'键")
                if not str(item["image"]).startswith("data:image;base64,"):
                    raise ValueError(f"参数类型错误:当'type'为'image'时，'image'值必须为base64格式的字符串")
            if item["type"] == "text" and "text" not in item:
                raise ValueError("参数类型错误:当'type'为'text'时，必须包含'text'键")
    try:
            torch.cuda.empty_cache()
            model_dir = _get_model(Model.VL)
            logging.info("视觉模型加载中...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype="auto", device_map="auto"
            )
            logging.info("视觉模型加载完毕...")

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
            logging.info("视觉模型推理中...")
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            logging.info("视觉模型推理完毕！")
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if output_text is not None:
                output_text = output_text[0]

            logging.info(f'''视觉模型推理完毕：
                @output_text:{output_text}''')
            return output_text
    except torch.cuda.OutOfMemoryError as e:
        raise MemoryError(f"model_call 异常: 视觉模型调调用过程显存不足，请检查是否使用了GPU：{e}")
    except Exception as e:
        raise Exception(f"model_call 异常: 视觉模型调用过程中出现异常：{e}")

def _call_tts(text:str)->bytes: 

    try:
        # 创建一个文本到语音的处理管道，指定任务类型为文本到语音，并使用指定的模型
        sambert_hifigan_tts = pipeline(task=Tasks.text_to_speech, model=Model.TTS.value)

        # 使用管道处理输入文本，指定使用的语音风格为 'zhitian_emo'
        output = sambert_hifigan_tts(input=text, voice='zhitian_emo')

        # 从输出中提取生成的 WAV 音频数据
        wav = output[OutputKeys.OUTPUT_WAV]

        return wav
    except Exception as e:
        raise Exception(f"model_call 异常: 文本到语音模型调用过程中出现异常：{e}")
