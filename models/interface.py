from enum import Enum
from typing import Generator, List, Dict
from numpy.typing import NDArray
from abc import ABC, abstractmethod

class Level(Enum):
    # CEFR levels
    A1 = 1
    A2 = 2
    B1 = 3
    B2 = 4
    C1 = 5
    C2 = 6


class ModelService(ABC):
    """Model service interface definition"""
    
    @abstractmethod
    def test_level(self, message_input: str) -> Level:
        """
        Desc:
            Return the level of the user, according to the composition wrote from the user.
            
        Usecase:
            >>> test_level("We have so many good things to eat for dinner.")
            Level.A1
        """
        pass

    @abstractmethod
    def get_img_info(self, img: str|NDArray, level: 'Level') -> Generator[str|dict,None,None]:
        """
            Desc:
                Return the information of the image, including a description of the image and the words within it along with corresponding bounding boxes, based on the user's level.
                
            Usecase:
                >>> get_img_info(img,Level.A1)
                {
                "desc":"A bird hit a laptop, spilling coffee from a cup. A mobilephone fell in the dustbin.",
                "translation":"一只鸟撞到了一台平板， 从杯子中溢出了咖啡, 一部手机掉进了垃圾箱。",
                "words":[
                    {"text":"laptop","location":[("123","456"),("126","467")],"translation":"平板"},
                    {"text":"bird","location":[("100","200"),("110","210")],"translation":"鸟"},
                    {"text":"coffee","location":[("300","400"),("310","410")],"translation":"咖啡"},
                    {"text":"cup","location":[("320","420"),("330","430")],"translation":"杯子"},
                    {"text":"mobilephone","location":[("500","600"),("510","610")],"translation":"手机"},
                    {"text":"dustbin","location":[("520","620"),("530","630")],"translation":"垃圾箱"}
                    ]
                }
        """
        pass

    @abstractmethod
    def get_conversation(self, conversation: List, level: 'Level', img: NDArray|str) -> str:
        """
            Desc:
                Return the generator of conversation, including the user's question and the assistant's answer, based on the user's level.
                
            Usecase:
                >>> chat = [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hello"},
                    {"role": "user", "content": "What do you see in the image?"}
                ]
                >>> get_chat(chat,Level.A1,img)
                (a generator of response string)
        """
        pass

    @abstractmethod
    def get_new_context(self, words: List[str], level: 'Level') -> List[str]:
        """
        Desc:
            Generate a new context description using the given words, based on the user's level.
            
        Usecase:
            >>> get_new_context(["laptop", "cup", "mobilephone"], Level.A1)
            "[There is a laptop, a cup, and a mobile phone.,中文翻译]"
        """
        pass

    @abstractmethod
    def get_audio(self, text: str) -> bytes:
        """
        Desc:
            Generate an audio file from the given text and return the local file path.
            The audio file will be saved in a specific folder with a unique name based on the current timestamp.
        Usecase:
            >>> audio_path = get_audio("今天天气很晴朗")
        """
        pass