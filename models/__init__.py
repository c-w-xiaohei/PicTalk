"""
PicTalk 核心模型服务模块

1. Level 枚举 - CEFR语言能力等级标准（A1-C2）
2. ModelService 抽象接口 - 定义模型服务核心功能
3. 默认实现实例 - ModelServiceDefaultImpl 的全局实例

"""


from .impl import ModelServiceDefaultImpl
from .interface import ModelService, Level

# 实例化具体实现并对外暴露
service: ModelService = ModelServiceDefaultImpl()

# 对外暴露Level枚举类
__all__ = ['service', 'Level']



