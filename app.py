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
    app.close() # 如果 `app` 存在，则关闭它
    os.environ["GRADIO_ROOT_PATH"] = f"/{os.environ['JUPYTER_NAME']}/proxy/7860"
except NameError:# 如果 `app` 不存在，捕获 NameError 异常，并忽略该异常
    ENV = "Prod"
except KeyError:# 环境变量不存在,不位于Notebook环境中
    ENV = "Prod"


"""
- 日志配置

"""

# 配置基础日志配置
if ENV == "Prod":
    logging.basicConfig(level=logging.DEBUG,format='%(levelname)s - %(asctime)s - %(funcName)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
else:  
    logging.basicConfig(filename=os.path.join(os.path.dirname(__file__),"pictalk.log"),level=logging.DEBUG,format='%(levelname)s - %(asctime)s - %(funcName)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('gradio')
logger.setLevel(logging.DEBUG) 


if __name__ == "__main__":
    try:
        interface = create_interface()
        interface.launch()
    except Exception as e:
        logging.error(e)
        raise e