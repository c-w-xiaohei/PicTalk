from enum import Enum
from typing import Tuple,Generator
from numpy.typing import NDArray

class Level(Enum):
    # CEFR levels
    A1 = 1
    A2 = 2
    B1 = 3
    B2 = 4
    C1 = 5
    C2 = 6
def test_level(message_input:str) -> Tuple[Level]:
    """
        Desc:
            Return the level of the user, according to the composition wrote from the user.
            
        Usecase:
            >>> test_level("We have somany good things to eat for dinner.")
            (Level.A1)
            
    """
    
    example:Tuple[Level] = (Level.A1)
    
    return example



def get_img_info(img:NDArray[any],level:Level) -> dict:
    """ 
        Desc:
            Return the information of the image, including a description of the image and the words within it along with corresponding bounding boxes, based on the user's level.
            
        Usecase:
            >>> get_img_info(img,Level.A1)
            {"desc":"Laptop bird dustbin cup coffee hit mobilephone" output:"A bird hit a laptop, spilling coffee from a cup. A mobilephone fell in the dustbin.","words":[("laptop",("123","456"),("126","467")),("cup",("53","534"),("86","486"))...]}

    """
    
    example:dict = {
    "desc":"The description of the image",
    "words":[
        ("word1",("x1","y1"),("x2","y2")),
        ("word2",("x1","y1"),("x2","y2")),
        ("word3",("x1","y1"),("x2","y2"))
        ]
    }
    
    return example

def get_conversation(conversation:list,level:Level) -> Generator[str,None,None]:
    """
        Desc:
            Return the generator of conversation, including the user's question and the assistant's answer, based on the user's level.
            
        Usecase:
            >>> chat = [
                {"role": "user", "content": "Hello"}, 
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "What do you see in the image?"}
            ]
            >>> get_chat(chat,Level.A1)
            (a generator of response string)
    """
    
    exmample_arr = [
        "Hello.","What","do","you","see?"
    ]    
    
    for s in exmample_arr:
        yield s

def get_new_context(words:list[str],level:Level) -> str:
    """
        Desc:
            Generate a new context description using the given words, based on the user's level.
            
        Usecase:
            >>> get_new_context(["laptop","cup","mobilephone"],Level.A1)
            "There is a laptop, a cup, and a mobile phone."
            
    """
    
    example:str = "There is a laptop, a cup, and a mobile phone."
    return example

    

