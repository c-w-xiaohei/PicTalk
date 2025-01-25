GenNewText:str = """
你是一个作家。你需要根据给定的单词，生成简短的文本片段。例如：input:["laptop", "bird", "dustbin", "cup", "coffee", "hit", "mobilephone"],output:"While sitting at a cafe, a bird flew by and accidentally hit the laptop, causing the person to spill their coffee from the cup onto the table. Nearby, someone's mobilephone fell into the dustbin."
"""

def prompt_first():
    natural_message = "你是一名优秀的判别CERF英语等级的专家，现在请你根据下面的文段，判断改同学的英语等级，请直接告诉我这名同学的CERF等级是什么,例如'A1'，"
    return natural_message
"""
def prompt_second_first():
    natural_message = " 用中文生成输入图片内容的详细描述和图片中所有实体的描述列表。输出为格式为：{"global_caption":"详细描述", "caption_list":["实体A的描述", "实体B的描述", "实体C的描述", ...直到所有实体都被描述完]}。"
    return natural_message
"""