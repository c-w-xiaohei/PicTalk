# Judgement
def prompt_judge():
    natural_message = "你是一名优秀的判别CERF英语等级的专家，现在请你根据下面的文段，判断改同学的英语等级，请直接告诉我这名同学的CERF等级是什么,例如'A1'，"
    return natural_message

# Image: Part 1
def prompt_descirbe_image():
    natural_message = '描述图片,输出json格式的字符串，{"global_caption":"详细描述", "caption_list":["实体A的描述", "实体B的描述", "实体C的描述", ...直到所有实体都被描述完]}'
    return natural_message 

# Image: Part 2
def prompt_extract_system():
    natural_message = '''你是一名归纳师以及英语教师。
    你需要根据**描述列表**和**CEFR英语等级**提取：
    1.实体单词列表(要求根据CEFR等级提取,简洁,尽量提取单个词汇,并配上中文翻译)
    2.场景描述片段(内容要求:根据CEFR等级,使用1中提取的单词,严格遵循描述列表中的描述,必要时辅以补充,并配上自然的中文翻译;文风要求:简短精炼,语言优美)
    最终输出 json 字符串,不带其他 json 格式之外的字符
    '''
    return natural_message

def prompt_extract_example():
    user = 'A1,{"global_caption":"三个穿着西装的人在跑道上跑步，背景是高楼和蓝天。", "caption_list":["三个穿着西装的人", "跑道", "高楼", "蓝天"]}'
    assistant = '''{"words":[["suits","西装"],["track","跑道"],["tall buildings","高楼"],["blue sky","蓝天"],["run","跑步"]],"desc_en":"Three people in suits are running on a track, with tall buildings and a blue sky as the backdrop.","desc_cn":"三个穿着西装的人在跑道上跑步，背后是高楼和蓝天。"}'''

    return [{"role":"user","content":user},{"role":"assistant","content":assistant}]

# Image: Part 3
def prompt_bbox():
    natural_message = '''print the bboxes of input objects in a format of 
    '[['objA',('100','200'),('560','665')],['objB',('456','565'),('788','989')],...untill no objects left]'
    '''
    return natural_message

# Conversation
def prompt_translate():
    natural_message = '你是一名专业的英文翻译专家，现在请你根据CEFA水平，将接受到的文段翻译为对应水平的英文。如：input：A1，一只鸟撞上了笔记本电脑，结果杯子里的咖啡洒了出来。同时，一部手机也掉进了垃圾桶。output: A bird hit a laptop, spilling coffee from a cup. A mobilephone fell in the dustbin.'
    return natural_message
# Context
def prompt_context():
    natural_message = '你是一名场景描述专家，现在请你根据单词列表，还有CERF英语等级，生成一段语句进行场景描述。如：input:"A2,laptop bird dustbin cup coffee hit mobilephone" output:"A bird hit a laptop, spilling coffee from a cup. A mobilephone fell in the dustbin.'
    return natural_message


