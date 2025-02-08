from numpy.typing import NDArray
import cv2
import base64
import json
import logging

logger = logging.getLogger("gradio")


def generate_image_html(words: list, current_image: NDArray) -> str:
    """生成包含图片及标注的HTML代码。

    参数:
        words (list): 包含标注信息的列表，每个元素为一个字典，包含以下键值：
            - "text" (str): 标注的文本内容。
            - "location" (list): 标注在图片中的位置，格式为[(x1, y1), (x2, y2)]，表示矩形框的左上角和右下角坐标。
            - "translation" (str): 标注文本的翻译内容。
        current_image (NDArray): 当前图片的NumPy数组表示。

    返回值:
        str: 生成的HTML代码，用于在网页中显示图片及其标注信息。
    """
    logger.debug(f"Frontend:生成图片及标注html代码中\n     @words: {words}\n")

    # 转换颜色空间到 BGR (OpenCV 默认)
    current_image_bgr = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)

    # 压缩图片
    # resized_image = cv2.resize(current_image_bgr, (400,400))
    resized_image = current_image_bgr
    _, buffer = cv2.imencode(".jpg", resized_image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    # 将压缩后的图片转为base64编码
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    img_src = f"data:image/png;base64,{img_base64}"

    # 将压缩后的图片转为base64编码
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    img_src = f"data:image/jpeg;base64,{img_base64}"

    # 生成单词badge，包含翻译内容
    badges = "".join(
        f'<div class="badge" title="{word["translation"]}">{word["text"]}</div>'
        for word in words
    )

    # 生成矩形框坐标数据，将千分比转换为实际像素值
    word_info = {
        word["text"]: {
            "coordinates": [
                float(word["location"][0][0]) / 1000,
                float(word["location"][0][1]) / 1000,
                float(word["location"][1][0]) / 1000,
                float(word["location"][1][1]) / 1000,
            ],
            "translation": word["translation"],
        }
        for word in words
        if word.get("text") and word.get("translation") and word.get("location")
    }
    word_info_json = json.dumps(word_info) if word_info else "{}"

    # 生成HTML代码
    html_content = f"""
    <div class="image-container">
    <div class="badges">{badges}</div>
    <img src="{img_src}" alt="Uploaded Image" id="uploaded-image" style="max-width: 100%; max-height: 100%;">
</div>
<script>
    const body = document.querySelector('body');
    body.style.overflow = 'hidden';
    const badges = document.querySelectorAll('.badge');
    const image = document.getElementById('uploaded-image');

    // 动态计算图片高度
    image.onload = function() {{
        const container = document.querySelector('.image-container');
        const badgesDiv = document.querySelector('.badges');
        const containerHeight = container.getBoundingClientRect().height;
        const badgesHeight = badgesDiv.getBoundingClientRect().height;
        image.style.height = `${{containerHeight - badgesHeight }}px`;
    }};

    let activeBoxes = [];
    const wordInfo = JSON.parse('{word_info_json}');

    badges.forEach(badge => {{
        badge.addEventListener('click', () => {{
            const word = badge.textContent;
            
            // Remove previous active boxes
            activeBoxes.forEach(box => box.remove());
            activeBoxes = [];

            // Find the bounding box for the word
            if (wordInfo[word]) {{
                const {{ coordinates, translation }} = wordInfo[word];
                const [x1, y1, x2, y2] = coordinates;
                const box = document.createElement('div');
                box.classList.add('bounding-box');

                // Add word and translation
                const label = document.createElement('div');
                label.classList.add('label');
                label.textContent = `${{word}} - ${{translation}}`;
                box.appendChild(label);

                const badgesDiv = document.querySelector('.badges');
                const badgesHeight = badgesDiv.getBoundingClientRect().height;
                // Set box styles
                box.style.left = `${{x1 * image.width}}px`;
                box.style.top = `${{y1 * image.height + 4 + badgesHeight}}px`;
                box.style.width = `${{(x2 - x1) * image.width}}px`;
                box.style.height = `${{(y2 - y1) * image.height}}px`;
                image.parentElement.appendChild(box);
                activeBoxes.push(box);
            }}
        }});
    }});
</script>
<style>
    .bounding-box {{
        position: absolute;
        border: 4px solid #FF5722;
        z-index: 10;
        background: rgba(255, 87, 34, 0.1);
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }}
    .label {{
        position: absolute;
        top: -30px;
        left: 0;
        background: #FF5722;
        color: white;
        padding: 6px 12px;
        font-size: 14px;
        font-weight: 500;
        border-radius: 4px;
        white-space: nowrap;
    }}
    .image-container {{
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        max-height: 100%;
    }}
    .badges {{
        margin-bottom: 10px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }}
    .badge {{
        display: inline-block;
        padding: 8px 16px;
        background-color: #007bff;
        color: white;
        cursor: pointer;
        border-radius: 16px;
        font-size: 14px;
        font-weight: 500;
        transition: background-color 0.3s ease;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1);
    }}
    .badge:hover {{
        background-color: #0056b3;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
    }}
</style>
    """
    # 将HTML内容转换为data URI
    html_encoded = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
    return f"""
    <iframe 
        src="data:text/html;base64,{html_encoded}" 
        style="width: 100%; height: 38vh; border: none;"
    ></iframe>
    """


def generate_badge_html(
    words: list,
) -> str:
    """生成包含图片及标注的HTML代码。

    参数:
        words (list): 包含标注信息的列表，每个元素为一个字典，包含以下键值：
            - "text" (str): 标注的文本内容。
            - "location" (list): 标注在图片中的位置，格式为[(x1, y1), (x2, y2)]，表示矩形框的左上角和右下角坐标。
            - "translation" (str): 标注文本的翻译内容。

    返回值:
        str: 生成的HTML代码，用于在网页中显示单词信息。
    """
    # 生成单词badge，包含翻译内容
    badges = "".join(f'<div class="badge">{word["text"]}</div>' for word in words)

    # 生成HTML代码
    return f"""
    <div class="badges">{badges}</div>
    <style>
        .badges {{
            margin-bottom: 10px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .badge {{
            display: inline-block;
            padding: 8px 16px;
            background-color: #007bff;
            color: white;
            cursor: pointer;
            border-radius: 16px;
            font-size: 14px;
            font-weight: 500;
            transition: background-color 0.3s ease;
            box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1);
        }}
        .badge:hover {{
            background-color: #0056b3;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        }}
    </style>
    """


def generate_context_list_html(contexts_list: list) -> str:
    """生成语境列表HTML

    Args:
        contexts_list: 一个语境列表，其中每个语境都是一个字典，包含 "en" (英文), "cn" (中文), 和 "audio" (音频路径) 键。
            例如: [{"en":"english!","cn":"中文！","audio":"path/to/audio"},...]

    Returns:
        包含语境列表的HTML字符串。
    """
    logger.debug(f"Frontend:生成语境列表中\n     @context_list: {contexts_list}")
    context_items = "".join(
        f"""
        <div class="context-item">
            <div class="context-text">
                <span class="english-text">{context["en"]}</span>
            </div>
            <div class="translation-text">
                {context["cn"]}
            </div>
            <button class="audio-button" onclick='new Audio("{context["audio"]}").play()'>🔊</button>
        </div>
        """
        for context in contexts_list
    )
    return (
        f"""
    <div class="context-list">
        {context_items}
    </div>
    """
        + r"""
    <style>
    .context-list {
        display: flex;
        flex-direction: column;
        gap: 12px;
        padding: 16px;
        background-color: var(--background-fill-primary);
        color: var(--body-text-color);
    }
    .context-item {
        background-color: var(--block-background-fill);
        border: 1px solid var(--border-color-primary);
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: background-color 0.2s ease-in-out;
    }
    .context-item:hover {
        background-color: var(--color-accent-soft);
    }
    .context-text {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
        font-weight: 500;
        color: var(--body-text-color);
    }
    .translation-text {
        color: var(--body-text-color-subdued);
        line-height: 1.6;
    }
    .audio-button {
        background-color: var(--block-label-background-fill);
        color: var(--link-text-color);
        border: none;
        border-radius: 6px;
        cursor: pointer;
        padding: 8px 12px;
        font-size: 14px;
        transition: background-color 0.2s ease-in-out, color 0.2s ease-in-out;
    }
    .audio-button:hover {
        background-color: var(--link-text-color-hover);
        color: var(--link-text-color-active);
    }
    .audio-button:focus {
        outline: none;
        box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.3);
    }
</style>
    """
    )


def generate_processing_html(step: str) -> str:
    """生成处理步骤的HTML提示信息。

    参数:
        step (str): 当前处理的步骤描述。

    返回值:
        str: 生成的HTML代码，用于提示用户当前处理步骤。
    """
    return (
        f"""
    <div class="processing-step">
        <p>{step}</p>
    </div>
    """
        + r"""
<style>
    .processing-step {
        background-color: var(--block-background-fill); 
        padding: 10px;
        border: 1px solid var(--block-border-color); 
        border-radius: 5px;
        margin: 10px 0;
        text-align: center;
        color: var(--body-text-color); 
    }
</style>
    """
    )


def test_generate_image_html():
    # 1. 读取图片并转换为NDArray
    image_path = "/mnt/workspace/test.png"
    current_image = cv2.imread(image_path)
    if current_image is None:
        raise FileNotFoundError(f"图片文件未找到：{image_path}")

    # 2. 定义标注信息
    words = [
        {
            "text": "laptop",
            "location": [("123", "456"), ("126", "467")],
            "translation": "平板",
        },
        {
            "text": "bird",
            "location": [("100", "200"), ("340", "260")],
            "translation": "鸟",
        },
        {
            "text": "coffee",
            "location": [("300", "400"), ("310", "410")],
            "translation": "咖啡",
        },
        {
            "text": "cup",
            "location": [("320", "420"), ("330", "430")],
            "translation": "杯子",
        },
        {
            "text": "mobilephone",
            "location": [("500", "600"), ("510", "610")],
            "translation": "手机",
        },
        {
            "text": "dustbin",
            "location": [("520", "620"), ("530", "630")],
            "translation": "垃圾箱",
        },
    ]

    # 3. 调用函数生成HTML
    html_output = generate_image_html(words, current_image)
    print("success!")
    with open("test.html", "w", encoding="utf-8") as file:
        file.write(html_output)


def test_context():
    """演示如何使用 generate_context_list_html 函数."""
    contexts = [
        {"en": "Hello, world!", "cn": "", "audio": "path/to/hello.mp3"},
        {"en": "How are you?", "cn": "", "audio": "path/to/how_are_you.mp3"},
        {"en": "Goodbye!", "cn": "", "audio": "path/to/goodbye.mp3"},
    ]

    html_output = generate_context_list_html(contexts)
    print("success!")
    with open("test2.html", "w", encoding="utf-8") as file:
        file.write(html_output)


if __name__ == "__main__":
    test_generate_image_html()
    test_context()
