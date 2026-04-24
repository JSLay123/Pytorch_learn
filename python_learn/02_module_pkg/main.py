# 1.模块--------------------------------------------------------------------------------------------------------
# 一个python文件就是一个模块，模块中定义的函数、变量等都可以被其他模块导入使用
import utils
res = utils.add(1, 2)

# 从utils模块中导入add函数
from utils import subtract
res = utils.subtract(1, 2)
print(res)

# 使用*号导入utils模块中的所有函数和变量
from utils import *
res = add(1, 2)
print(res)

# 从utils模块中导入add函数并重命名为my_add
from utils import add as my_add
import utils as ut
res = my_add(1, 2)
print(res)
res = ut.subtract(1, 2)
print(res)

# 2. Python的搜索路径-----------------------------------------------------------------------------------------------------------------
# 当我们导入一个模块时，Python会按照一定的顺序搜索模块文件，搜索路径包括当前目录、环境变量PYTHONPATH指定的目录以及Python安装目录等
# 保存在sys.path列表中，我们可以通过sys模块查看和修改搜索路径
# a.自动搜索当前路径，如果当前路径下有utils.py文件，可以直接import utils
# b.如果当前路径下没有utils.py文件，Python会继续搜索环境变量PYTHONPATH指定的目录，如果在这些目录中找到了utils.py文件，就会导入这个文件
# c.如果在环境变量PYTHONPATH指定的目录中也没有找到utils.py文件，Python会继续搜索Python安装目录下的标准库，如果在标准库中找到了utils.py文件，就会导入这个文件
# d.如果在以上路径中都没有找到utils.py文件，Python会抛出ModuleNotFoundError异常，提示找不到模块
import sys
print([path for path in sys.path])

# 3.__name__变量-----------------------------------------------------------------------------------------------------------------
# 每个模块都有一个__name__变量，表示模块的名字，当一个模块被直接运行时，__name__变量的值为'__main__'，
# 当一个模块被其他模块导入时，__name__变量的值为模块的名字
# 
print(__name__) # 输出'__main__'，表示当前模块是被直接运行的
print(utils.__name__) # 输出'utils'，表示utils模块的名字
# 主程序入口
# 因此，在运行一个模块时，我们可以通过判断__name__变量的值来决定是否执行某些代码，例如：
if __name__ == '__main__':
    print("当前模块是被直接运行的")


# 4.包---------------------------------------------------------------------------------------------------------------------------
# 包是一个包含多个模块的目录，包中必须有一个__init__.py文件，表示这个目录是一个包，包中的模块可以通过包名.模块名的方式导入使用，例如：

# 从包中导入模块，然后 模块名.函数名 的方式调用函数
# from my_package import utils

# 从包中导入模块中的函数
# from my_package.utils import add

# 包中包：from my_package.subpackage import module