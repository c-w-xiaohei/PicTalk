from frontend import create_interface
import os
import logging
from logging.handlers import MemoryHandler

"""
- 环境配置

"""
ENV:str= "Dev"
try:
    # Notebook 配置
    os.environ["GRADIO_ROOT_PATH"] = f"/{os.environ['JUPYTER_NAME']}/proxy/7860"
except KeyError:# 环境变量不存在,不位于Notebook环境中
    ENV = "Prod"


"""
- 日志配置

"""
logger = logging.getLogger('gradio')
logger.setLevel(logging.INFO) 
formatter = logging.Formatter(fmt='%(levelname)s - %(asctime)s - %(filename)s - %(funcName)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 配置基础日志配置
if ENV == "Dev" :
    handler = logging.FileHandler(filename=os.path.join(os.path.dirname(__file__),"pictalk.log"),encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


if __name__ == "__main__":
    try:
        interface = create_interface()
        logger.info("Gradio 实例启动成功！")
        interface.launch()
    except Exception as e:
        logger.error(e)
        raise e