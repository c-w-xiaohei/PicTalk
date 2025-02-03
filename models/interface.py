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
    def get_img_info(self, img: NDArray, level: 'Level') -> Dict:
        """
            Desc:
                Return the information of the image, including a description of the image and the words within it along with corresponding bounding boxes, based on the user's level.
                img: base64 encoded image
                
            Usecase:
                >>> get_img_info(img,Level.A1)
                {
                "desc":"A bird hit a laptop, spilling coffee from a cup. A mobilephone fell in the dustbin.",
                "translation":"一只鸟撞到了一台平板， 从杯子中溢出了咖啡, 一部手机掉进了垃圾箱。",
                "words":[("laptop",("123","456"),("126","467")),("cup",("53","534"),("86","486"))...],
                "words_translation":["平板","杯子"...]
                }
        """
        pass

    @abstractmethod
    def get_conversation(self, conversation: List, level: 'Level', img: NDArray) -> Generator[str, None, None]:
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
    def get_new_context(self, words: List[str], level: 'Level') -> str:
        """
        Desc:
            Generate a new context description using the given words, based on the user's level.
            
        Usecase:
            >>> get_new_context(["laptop", "cup", "mobilephone"], Level.A1)
            "There is a laptop, a cup, and a mobile phone."
        """
        pass

    @abstractmethod
    def get_audio(self, text: str) -> str:
        """
        Desc:
            Generate an audio file from the given text and return the local file path.
            The audio file will be saved in a specific folder with a unique name based on the current timestamp.
        Usecase:
            >>> audio_path = get_audio("今天天气很晴朗")
            >>> print(audio_path)
            "/path/to/audio/20250122_123456.wav"
        """
        pass