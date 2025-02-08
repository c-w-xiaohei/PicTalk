import logging
from logging.handlers import MemoryHandler
from aspect import exception_to_logs, log_error
import gradio as gr
from gradio.route_utils import get_root_url
from gradio.processing_utils import (
    save_bytes_to_cache,
    hash_bytes,
    save_img_array_to_cache,
)
from models import service, Level
from typing import List, Generator, Optional, Dict
from PIL import Image
from numpy.typing import NDArray
from numpy import frombuffer, uint8
import cv2
from html_utils import (
    generate_context_list_html,
    generate_image_html,
    generate_badge_html,
    generate_processing_html,
)
from os import getenv


# 引入抽象类

# 测试语句
# from test_frontend import MockModelService
# service = MockModelService()

"""
- 日志配置

"""
logger = logging.getLogger("gradio")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(levelname)s - %(asctime)s - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console = logging.StreamHandler()

console.setLevel(logging.INFO)
console.setFormatter(formatter)

memory_handler = MemoryHandler(
    capacity=1000, flushLevel=logging.ERROR, target=console, flushOnClose=False
)
memory_handler.setFormatter(formatter)
logger.addHandler(memory_handler)


# 配置字典
level_config = {
    "高考成绩（全国卷）": {
        "choices": [
            "42.5%~50%: A1/A2",
            "57.5%~65%: B1",
            "72.5%~87.5%: B2",
            "90%~97.5%: C1/C2",
        ],
        "default": "42.5%~50%: A1/A2",
    },
    "CEFR": {"choices": ["A1", "A2", "B1", "B2", "C1", "C2"], "default": "A1"},
}


class StateType:
    def __init__(self):
        self.current_image: Optional[NDArray] = None
        self.current_words: List[dict] = []
        self.chat_history: List[dict] = []
        self.context_list: List[dict] = []
        self.current_level: Optional[Level] = Level.A1
        self.image_url: Optional[str] = None


# 定义界面布局
def create_interface():
    with gr.Blocks() as demo:
        # 初始化 State 对象
        state: StateType = gr.State(StateType())

        with gr.Row():
            # 左侧控制面板
            with gr.Column(scale=3):
                # 水平选择区域
                with gr.Column():
                    level_standard = gr.Radio(
                        choices=["高考成绩（全国卷）", "CEFR"],
                        label="水平选择标准",
                        value="高考成绩（全国卷）",
                    )
                    level_select = gr.Dropdown(
                        label="选择英语水平",
                        choices=level_config["高考成绩（全国卷）"][
                            "choices"
                        ],  # 默认使用高考成绩的选项
                        value=level_config["高考成绩（全国卷）"]["default"],  # 默认值
                        interactive=True,
                    )
                    level_test_input = gr.Textbox(
                        label="输入英语写作片段检测水平", lines=8, max_length=300
                    )
                    test_button = gr.Button("检测")

                # 图片上传区域
                with gr.Column():
                    image_input = gr.Image(
                        label="上传图片", type="numpy"  # 直接返回 numpy.ndarray
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
                                show_label=False,
                            )

                    with gr.Tab("新语境"):
                        with gr.Column():
                            # 单词badge区域
                            with gr.Row():
                                with gr.Column(scale=8):
                                    word_badges = gr.HTML("<h1>请上传图片</h1>")
                                with gr.Column(scale=1):
                                    word_input = gr.Textbox(
                                        label="输入单词", placeholder="输入单词后按回车"
                                    )
                                    context_button = gr.Button("生成语境")
                            # 语境列表
                            context_list = gr.HTML("")

        # 交互逻辑
        msg.submit(
            fn=generate_conversation,
            inputs=[chatbot, msg, state],
            outputs=[chatbot, msg],
            api_name="chat_answer",
        )

        level_standard.change(
            fn=lambda x: gr.Dropdown(
                choices=level_config[x]["choices"], value=level_config[x]["default"]
            ),
            inputs=level_standard,
            outputs=level_select,
        )
        test_button.click(
            fn=test_level,
            inputs=[level_test_input, level_standard, state],
            outputs=level_select,
        )

        def _handle_image_input(
            image_input: NDArray, state: StateType, req: gr.Request
        ):
            store_img(image_input, demo, req, state)
            for i in process_image(state):
                yield i

        image_input.change(
            fn=_handle_image_input,
            inputs=[image_input, state],
            outputs=[image_display, desc_en, desc_cn, word_badges],
        )

        word_input.submit(
            fn=lambda word, _, state: generate_word_badges(word, state),
            inputs=[word_input, word_badges, state],
            outputs=word_badges,
        )

        def _handel_context_button(state: gr.State, req: gr.Request):
            for i in generate_new_context("", demo, req, state):
                yield i

        context_button.click(
            fn=_handel_context_button, inputs=state, outputs=context_list
        )
        demo.unload(lambda: memory_handler.buffer.clear())
    return demo


@exception_to_logs(logger, custom_message="测试水平时出现错误")
def test_level(text: str, standard: str, state: StateType) -> str:
    """测试用户英语水平"""
    logger.info("Frontend: 开始测试用户英语水平")
    level = service.test_level(text)
    state.current_level = level
    logger.info(f"Frontend: 测试结果 - {level}")
    return _format_level(level, standard)


@exception_to_logs(logger, custom_message="压缩图片时出现错误")
def compress_image(state: StateType) -> None:
    if state.current_image is None:
        return
    # 若图片长宽大于1000px则压缩减半
    image = state.current_image
    while image.shape[0] > 1000 or image.shape[1] > 1000:
        image = cv2.resize(
            image,
            (image.shape[1] // 2, image.shape[0] // 2),
            interpolation=cv2.INTER_AREA,
        )
    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    # 将编码后的字节数据转换回原始图像
    image_array = frombuffer(buffer, dtype=uint8)
    state.current_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    logger.info("图片压缩成功！")


def process_image(state: StateType):
    """处理上传的图片"""
    try:
        if state.current_image is None:
            yield "<h1>请上传图片</h1>", "", "", ""
            return

        logger.info("Frontend: 开始处理图片")
        logger.info("----------------------------")

        logger.info(state.current_level)
        result_gen = service.get_img_info(
            state.image_url if getenv("ACCESS_TOKEN", "") else state.current_image,
            state.current_level,
        )
        for step in result_gen:
            if step == "<END>":
                break
            yield generate_processing_html(step), "", "", ""
        result = next(result_gen)
        logger.info("----------------------------")
        logger.info(f"Frontend: 图片处理完成\n处理结果 - {result}")

        state.current_words = result["words"]

        # 生成HTML显示内容
        logger.info("Frontend: 生成图片显示HTML")
        html_content = generate_image_html(result["words"], state.current_image)
        logger.info("result - " + html_content[:300] + "...")

        # 生成单词badge
        badges = generate_word_badges("", state)

        # 对 desc 和 translation 中的单词进行 Markdown 加粗
        desc = result["desc"]
        translation = result["translation"]
        for word_data in result["words"]:
            word = word_data["text"]
            desc = desc.replace(word, f"**{word}**")
            translation = translation.replace(
                word_data["translation"], f"**{word_data['translation']}**"
            )

        yield html_content, desc, translation, badges
    except Exception as e:
        log_error(logger, e, "处理图片时出现错误")
        raise


def generate_conversation(
    chat_history: List, msg: str, state: StateType
) -> Generator[str, None, None]:
    """生成对话"""
    try:
        if state.current_image is None:
            logger.warning("Frontend: 未上传图片，无法生成对话")
            yield [{"role": "assistant", "content": "请先上传图片"}], ""
            return

        logger.info("Frontend: 开始生成对话")
        logger.info("----------------------------")
        if str(chat_history) == str([{"role": "assistant", "content": "请先上传图片"}]):
            chat_history = []
        if msg:
            chat_history.append({"role": "user", "content": msg})
        logger.info(f"@chat_history: {chat_history}")
        # 直接使用当前图像数组
        img = state.current_image
        # 调用模型服务并处理流式输出
        streamer = service.get_conversation(chat_history, state.current_level, img)
        chat_history.append({"role": "assistant", "content": ""})
        for chunk in streamer:
            chat_history[-1]["content"] += chunk
            yield chat_history, ""
        logger.info("----------------------------")
        logger.info("Frontend: 对话生成完成")
        return
    except Exception as e:
        log_error(logger, e, "生成对话时出现错误")
        raise


@exception_to_logs(logger, custom_message="生成单词标签时出现错误")
def generate_word_badges(input_word: str, state: StateType) -> str:
    """生成单词标签"""
    logger.info("Frontend: 开始生成单词标签")
    if input_word:
        state.current_words.append({"text": input_word})
    badges = generate_badge_html(state.current_words)
    logger.info(f"Frontend: 单词标签生成完成 - {badges[:100]}")
    return badges


def generate_new_context(
    word: str, demo: gr.Blocks, request: gr.Request, state: StateType
):
    """生成新语境"""
    try:
        logger.info("Frontend: 开始生成新语境")
        logger.info("----------------------------")
        # 提示用户正在处理单词列表
        if state.current_image is None:
            yield "<h1>未上传图片</h1>"
            return

        text_list = [w.get("text") for w in state.current_words if w.get("text")]
        if word:
            text_list.append(word)

        # 提示用户正在获取新语境
        yield generate_processing_html("(0/2) - 正在获取新语境...")
        context:List[str] = service.get_new_context(text_list, state.current_level)
        yield generate_processing_html("(1/2) - 正在获取音频...")
        path = get_audio(context[0], demo, request)
        state.context_list.append({"en": context[0], "cn": context[1], "audio": path})
        html_content = generate_context_list_html(state.context_list)
        logger.info("----------------------------")
        logger.info(f"Frontend: 新语境生成完成 - {html_content}")
        yield html_content
    except Exception as e:
        log_error(logger, e, "生成新语境时出现错误")
        raise


@exception_to_logs(logger, custom_message="获取音频时出现错误")
def get_audio(text: str, demo: gr.Blocks, request: gr.Request):
    logging.info("Frontend:开始获取音频")
    root = get_root_url(
        request=request, route_path="/gradio_api/queue/join", root_path=demo.root_path
    )
    wav = service.get_audio(text)
    path = save_bytes_to_cache(wav, "audio.wav", demo.GRADIO_CACHE)
    # 更新音频和图片URL
    url = f"{root}/gradio_api/file={path}"
    return url


@exception_to_logs(logger, custom_message="存储图像时出现错误")
def store_img(img: NDArray, demo: gr.Blocks, request: gr.Request, state: StateType):
    if img is None:
        state.chat_history = state.current_words = state.context_list = []
        state.current_image = state.image_url = None
        return
    state.current_image = img
    compress_image(state)
    root = get_root_url(
        request=request, route_path="/gradio_api/queue/join", root_path=demo.root_path
    )
    path = save_img_array_to_cache(state.current_image, demo.GRADIO_CACHE, "jpeg")
    # 更新音频和图片URL
    url = f"{root}/gradio_api/file={path}"
    state.image_url = url


def _format_level(level: Level, standard: str) -> str:
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
