GenNewText:str = """
你是一个作家。你需要根据给定的单词，生成简短的文本片段。例如：input:["laptop", "bird", "dustbin", "cup", "coffee", "hit", "mobilephone"],output:"While sitting at a cafe, a bird flew by and accidentally hit the laptop, causing the person to spill their coffee from the cup onto the table. Nearby, someone's mobilephone fell into the dustbin."
"""

def prompt_first():
    natural_message = "你是一名优秀的判别CERF英语等级的专家，现在请你根据下面的文段，判断改同学的英语等级，请直接告诉我这名同学的CERF等级是什么,例如'A1'，"
    return natural_message

def prompt_second_first():
    natural_message = '用中文生成输入图片内容的详细描述和图片中所有实体的描述列表。输出json格式,例如：{"global_caption":"详细描述", "caption_list":["实体A的描述", "实体B的描述", "实体C的描述", ...直到所有实体都被描述完]}'
    return natural_message 

def prompt_second_fix():
    natural_message = '你是一名提取专家，请你根据描述列表提取单词列表,并输出json格式，例如，input:{"global_caption":"详细描述", "caption_list":["实体A的描述", "实体B的描述", "实体C的描述", ...直到所有实体都被描述完]},output:["实体A","实体B","实体C",...直至所有的实体全部输出]'
    return natural_message


def prompt_second_second():
    natural_message = '你是一个英语教学专家。你需要根据CERF英语等级，如“A1,A2”，根据给定的描述列表中的描述对象生成等级对应的英语片段,可以使用同义词，但不能改变原意，并配上自然的中文翻译。如：input:"A2,laptop bird dustbin cup coffee hit mobilephone" output:"A bird hit a laptop, spilling coffee from a cup. A mobilephone fell in the dustbin.一只鸟撞上了笔记本电脑，结果杯子里的咖啡洒了出来。同时，一部手机也掉进了垃圾桶。'
    return natural_message

def prompt_second_third():
    natural_message = '你是一个归纳专家以及英语教师。你需要从上述叙述片段中提取出对象要素的英文单词，要求：所提取的对象为名词，例如:input:"{"global_caption":"三个穿着西装的人在跑道上跑步，背景是高楼和蓝天。", "caption_list":["三个穿着西装的人", "跑道", "高楼", "蓝天"]}",output:"["suits ","track ","tall buildings","background","blue sky","run"]"注：尽量提取单个词汇'
    return natural_message


def prompt_second_forth():
    natural_message = 'print the bboxs of input objects in a format of output format:input objects:["blue shirt","white shorts","football player","football","goal","grass"],output format:[[objectA,(x1,y1),(x2,y2)],[objectB,(x1,y1),(x2,y2)],...untill no objects left]", "跑道", "高楼", "蓝天"]}",output:"["suits ","track ","tall buildings","background","blue sky","run"]"注：尽量提取单个词汇,输入内容为：'
    return natural_message

def prompt_context():
    natural_message = '你是一名场景描述专家，现在请你根据单词列表，还有CERF英语等级，生成一段语句进行场景描述。如：input:"A2,laptop bird dustbin cup coffee hit mobilephone" output:"A bird hit a laptop, spilling coffee from a cup. A mobilephone fell in the dustbin.'
    return natural_message

def prompt_translate():
    natural_message = '你是一名专业的英文翻译专家，现在请你根据CEFA水平，将接受到的文段翻译为对应水平的英文。如：input：A1，一只鸟撞上了笔记本电脑，结果杯子里的咖啡洒了出来。同时，一部手机也掉进了垃圾桶。output: A bird hit a laptop, spilling coffee from a cup. A mobilephone fell in the dustbin.'
    return natural_message

def prompt_contract():
    natural_message = '你现在是一名图片分析专家，请你根据下面的对话或者对话历史以及图片，生成回答的内容'
    return natural_message