from frontend import create_interface
import os
import logging

os.environ["GRADIO_ROOT_PATH"] = f"/{os.environ['JUPYTER_NAME']}/proxy/7860"

"""
- 日志配置

"""
logging.basicConfig(filename=os.path.join(os.path.dirname(__file__),"pictalk.log"), filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

if __name__ == "__main__":
    try:
        interface = create_interface()
        interface.launch()
    except Exception as e:
        logging.error(e)
        raise e