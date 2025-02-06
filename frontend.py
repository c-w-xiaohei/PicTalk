import logging
import gradio as gr
from gradio.route_utils import get_root_url
from gradio.processing_utils import save_bytes_to_cache,hash_bytes
from models import service, Level
from models import Level
from typing import Dict, List, Generator,Tuple
import base64
import os
from PIL import Image
from numpy.typing import NDArray
import cv2
from html_utils import  generate_context_list_html,generate_image_html,generate_badge_html

# 测试语句
# from test_frontend import MockModelService
# service = MockModelService()

# 配置字典
level_config = {
    "高考成绩（全国卷）": {
        "choices": ["42.5%~50%: A1/A2", "57.5%~65%: B1", "72.5%~87.5%: B2", "90%~97.5%: C1/C2"],
        "default": "72.5%~87.5%: B2"
    },
    "CEFR": {
        "choices": ["A1", "A2", "B1", "B2", "C1", "C2"],
        "default": "A1"
    }
}

class PicTalkApp:
    def __init__(self):
        self.current_level = Level.A1
        self.current_image:NDArray = None
        self.current_words = [] # eg: [{"text": "laptop", "location": [(10,150),(560,500)], "translation": "平板"},...]
        self.context_list = [] # eg: [{"en":"english!","cn":"中文！","audio":"path/to/audio"},...]
    
    def test_level(self, text: str, standard: str) -> str:
        """测试用户英语水平"""
        logging.info("Frontend: 开始测试用户英语水平")
        level = service.test_level(text)
        self.current_level = level
        logging.info(f"Frontend: 测试结果 - {level}")
        return self._format_level(level, standard)
            
    def process_image(self, image: NDArray, level: str) -> tuple:
        """处理上传的图片"""
        if image is None:
            logging.warning("Frontend: 未上传图片")
            return "<h1>请上传图片</h3>","","",""

        logging.info("Frontend: 开始处理图片")
        logging.info("----------------------------")
        img_array = image
        result = service.get_img_info(img_array, self.current_level)
        logging.info("----------------------------")
        logging.info(f"Frontend: 图片处理完成\n处理结果 - {result}")

        self.current_image = image
        self.current_words = result["words"]

        # 生成HTML显示内容
        logging.info("Frontend: 生成图片显示HTML")
        html_content = generate_image_html(result["words"], self.current_image)
        logging.info("result - " + html_content[:300] + "...")

        # 生成单词badge
        badges = self.generate_word_badges("")

        # 对 desc 和 translation 中的单词进行 Markdown 加粗
        desc = result["desc"]
        translation = result["translation"]
        for word_data in result["words"]:
            word = word_data["text"]
            desc = desc.replace(word, f"**{word}**")
            translation = translation.replace(word_data["translation"], f"**{word_data['translation']}**")


        return html_content, desc, translation, badges
        
    
    def generate_conversation(self, chat_history: List,msg:str) -> Generator[str, None, None]:
        """生成对话"""
        if self.current_image is None:
            logging.warning("Frontend: 未上传图片，无法生成对话")
            yield {"role": "assistant", "content": "请先上传图片"}
            return
        
        logging.info("Frontend: 开始生成对话")
        logging.info("----------------------------")
        if msg:
            chat_history.append({"role": "user", "content": msg})
        logging.info(f"@chat_history: {chat_history}")
        # 直接使用当前图像数组
        img = self.current_image
        # 调用模型服务并处理流式输出
        streamer = service.get_conversation(chat_history, self.current_level, img)
        chat_history.append({"role": "assistant", "content": ""})
        for chunk in streamer:
            chat_history[-1]["content"] += chunk
            yield chat_history
        logging.info("----------------------------")
        logging.info("Frontend: 对话生成完成")

    def generate_word_badges(self, input_word:str) -> str:
        """生成单词标签"""
        logging.info("Frontend: 开始生成单词标签")
        if input_word:
            self.current_words.append({"text":input_word})
        badges = generate_badge_html(self.current_words)
        logging.info(f"Frontend: 单词标签生成完成 - {badges[:100]}")
        return badges
    
    def generate_new_context(self, word: str,demo:gr.Blocks,request:gr.Request) -> Dict:
        """生成新语境"""
        logging.info("Frontend: 开始生成新语境")
        logging.info("----------------------------")
        text_list = [w.get("text") for w in self.current_words if w.get("text")]
        if word:
            text_list.append(word)
        context = service.get_new_context(text_list, self.current_level)
        path = self._get_audio(context,demo,request)
        self.context_list.append({"en":context,"cn":"","audio":path})
        html_content = generate_context_list_html(self.context_list)
        logging.info("----------------------------")
        logging.info(f"Frontend: 新语境生成完成 - {html_content}")
        return html_content
    
    def _get_audio(self,text:str,demo:gr.Blocks,request: gr.Request):
        logging.info("Frontend:开始获取音频")
        root = get_root_url(
            request=request, route_path="/gradio_api/queue/join", root_path=demo.root_path
        )   
        wav = service.get_audio(text)
        path = save_bytes_to_cache(wav,"audio.wav",demo.GRADIO_CACHE)
    # 更新音频和图片URL
        url = f"{root}/gradio_api/file={path}"
        return url
    
    def _format_level(self, level: Level, standard: str) -> str:
        """格式化水平显示"""
        formatted_level = level.name
        if standard == "高考成绩（全国卷）":
            if level == Level.A2 or level == Level.A1:
                formatted_level = "42.5%~50%: A1/A2"
            elif level == Level.B1:
                formatted_level = "57.5%~65%: B1"
            elif level == Level.B2:
                formatted_level = "72.5%~87.5%: B2"
            elif level == Level.C1 or level == Level.C2:
                formatted_level = "90%~97.5%: C1/C2"
        logging.info(f"Frontend: 格式化水平显示 - {formatted_level}")
        return formatted_level

# 创建应用实例
app = PicTalkApp()

# 定义界面布局
def create_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            # 左侧控制面板
            with gr.Column(scale=3):
                # 水平选择区域
                with gr.Column():
                    level_standard = gr.Radio(
                        choices=["高考成绩（全国卷）", "CEFR"],
                        label="水平选择标准",
                        value="高考成绩（全国卷）"
                    )
                    level_select = gr.Dropdown(
                        label="选择英语水平",
                        choices=level_config["高考成绩（全国卷）"]["choices"],  # 默认使用高考成绩的选项
                        value=level_config["高考成绩（全国卷）"]["default"],  # 默认值
                        interactive=True
                    )
                    level_test_input = gr.Textbox(
                        label="输入英语写作片段检测水平",
                        lines=8,
                        max_length=300
                    )
                    test_button = gr.Button("检测")
                
                # 图片上传区域
                with gr.Column():
                    image_input = gr.Image(
                        label="上传图片",
                        type="numpy"  # 直接返回 numpy.ndarray
                    )
            
            # 右侧主展示区域
            with gr.Column(scale=7):
                # 图片显示区域
                with gr.Column():
                    image_display = gr.HTML("<h1>请上传图片</h1>")
                
                # 互动功能区
                with gr.Tabs():
                    with gr.Tab("描述语段"):
                        with gr.Column():
                            desc_en = gr.Markdown()
                            desc_cn = gr.Markdown()
                    
                    with gr.Tab("对话"):
                        chatbot = gr.Chatbot(label="ChatBot", type="messages")
                        with gr.Row():
                            msg = gr.Textbox(
                                label="输入消息",
                                placeholder="输入消息后按回车",
                                show_label=False
                            )
                    
                    with gr.Tab("新语境"):
                        with gr.Column():
                            # 单词badge区域
                            with gr.Row():
                                with gr.Column(scale=8):
                                    word_badges = gr.HTML()
                                with gr.Column(scale=1):
                                    word_input = gr.Textbox(
                                        label="输入单词",
                                        placeholder="输入单词后按回车"
                                    )
                                    context_button = gr.Button("生成语境")
                            # 语境列表
                            context_list = gr.HTML()
        
        # 交互逻辑
        msg.submit(
            fn=app.generate_conversation,
            inputs=[chatbot, msg],
            outputs=chatbot,
            api_name="chat_answer"
        )

        level_standard.change(
            fn=lambda x: gr.Dropdown(
                choices=level_config[x]["choices"],
                value=level_config[x]["default"]
            ),
            inputs=level_standard,
            outputs=level_select
        )
        test_button.click(
            fn=app.test_level,
            inputs=[level_test_input, level_standard],
            outputs=level_select
        )
        
        image_input.change(
            fn=app.process_image,
            inputs=[image_input, level_select],
            outputs=[image_display,desc_en, desc_cn,word_badges]
        )
        
        word_input.submit(
            fn=lambda word, _: app.generate_word_badges(word),
            inputs=[word_input, word_badges],
            outputs=word_badges
        )
        
        def _handel_context_button(req:gr.Request):
            return app.generate_new_context("",demo,req)
        
        context_button.click(
            fn=_handel_context_button,
            outputs=context_list
        )
    
    return demo