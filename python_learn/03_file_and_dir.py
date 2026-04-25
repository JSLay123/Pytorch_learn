# 1.open()函数-----------------------------------------------
# open()函数用于打开一个文件，并返回一个文件对象
# open()函数的语法：open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)
    # file：要打开的文件路径，可以是相对路径或绝对路径
    # mode：打开文件的模式，默认为'r'，表示只读模式。
        # 常用的模式还有'w'（写入模式）、'a'（追加模式）、'r+'（读写模式）、'x'（创建模式）等
    # buffering：缓冲策略，默认为-1，表示使用系统默认的缓冲策略。可以设置为0（无缓冲）、1（行缓冲）或大于1的整数（固定大小的缓冲）
    # encoding：文件的编码方式，默认为None，表示使用系统默认的编码
    # errors：错误处理方式，默认为None，表示使用系统默认的错误处理方式
    # newline：控制换行符的处理方式，默认为None，表示使用系统默认的换行符处理方式
    # closefd：如果file是一个文件描述符，closefd参数指定是否在文件对象关闭时关闭文件描述符，默认为True
    # opener：一个可选的自定义打开器，可以用来替代内置的open()函数


# 2.文件对象的方法-----------------------------------------------
# 文件对象的方法有很多，常用的有：
# read(size=-1)：从文件中读取指定大小的内容，默认为-1，表示读取整个文件
# readline(size=-1)：从文件中读取一行内容，默认为-1，表示读取整行，连续调用可以自动读取多行
file_instance = open("/home/silei/WorkSpace_git/Pytorch_learn/python_learn/01_start.py", "r")
while True:
    line = file_instance.readline() # 读取一行内容
    if not line: # 如果读取到文件末尾，line将为一个空字符串
        break
    print(line.strip()) # 打印读取到的行内容，使用strip()去除行末的换行符
file_instance.close() # 关闭文件对象，释放资源

# readlines()：从文件中读取所有行内容，并返回一个列表

# write(string)：向文件中写入字符串; 
# 'w'模式会覆盖原有内容（不存在则创建，存在则覆盖）
# 'a'模式会在原有内容基础上追加内容,如果文件不存在会创建新文件
file_instance = open("/home/silei/WorkSpace_git/Pytorch_learn/python_learn/03_file.txt", "a")
file_instance.write("Hello, World!\n") # 向文件中写入一行内容
file_instance.write("This is a test file.\n") # 向文件中写入另一行内容

# writelines(lines)：向文件中写入一个字符串列表
file_instance.writelines(["Line 1\n", "Line 2\n", "Line 3\n"]) # 向文件中写入多行内容
file_instance.close() # 关闭文件对象，释放资源

# close()：关闭文件对象，释放资源

# flush()：刷新文件对象的缓冲区，将缓冲区中的内容写入文件
# seek(offset, whence=0)：移动文件指针到指定位置，offset表示偏移量，whence表示参考位置，默认为0（文件开头），1（当前位置），2（文件末尾）
# tell()：返回当前文件指针的位置
# with语句：使用with语句可以自动管理文件对象的上下文，确保文件在使用完毕后被正确关闭


# 3.os模块----------------------------------------------------------------------
# os模块提供了与操作系统进行交互的功能，可以用于文件和目录的操作
import os  
# os.path模块提供了用于处理文件路径的函数
# os.exists(path)：判断指定路径是否存在
print(os.path.exists("/home/silei/WorkSpace_git/Pytorch_learn/python_learn/03_file.txt")) # True
print(os.path.exists("/home/silei/WorkSpace_git/Pytorch_learn/python_learn/04_file.txt")) # False

# os.path.isfile(path)：判断指定路径是否是一个文件
# 区分文件和目录
print(os.path.isfile("/home/silei/WorkSpace_git/Pytorch_learn/python_learn/03_file.txt")) # True
print(os.path.isfile("/home/silei/WorkSpace_git/Pytorch_learn/python_learn")) # False

# os.path.isdir(path)：判断指定路径是否是一个目录

# os.path.join(path, *paths)：将多个路径组合成一个路径
print(os.path.join("/home/silei/WorkSpace_git/Pytorch_learn/python_learn", "03_file.txt"))

# os.path.split(path)：将路径分割成目录和文件名两部分，返回一个元组
# os.path.splitext(path)：将路径分割成文件名和扩展名两部分，返回一个元组
# os.listdir(path)：返回指定目录下的所有文件和目录的列表
print(os.listdir("/home/silei/WorkSpace_git/Pytorch_learn/python_learn"))   # ['01_start.py', '02_function.py', '03_file.txt']

# os.rename(src, dst)：重命名文件或目录
# os.remove(path)：删除指定路径的文件
# os.rmdir(path)：删除指定路径的空目录
# os.makedirs(path)：递归创建目录

# 4.pathlib模块----------------------------------------------------------------------
# pathlib模块提供了面向对象的文件系统路径操作方法
from pathlib import Path
# Path类表示一个文件系统路径，可以使用它来进行路径操作

# Path(path)：创建一个Path对象，path可以是一个字符串路径或另一个Path对象
path_instance = Path("/home/silei/WorkSpace_git/Pytorch_learn/python_learn/03_file.txt")
print(path_instance) # /home/silei/WorkSpace_git/python_learn/03_file.txt

# Path.exists()：判断路径是否存在
print(path_instance.exists()) # True
# Path.is_file()：判断路径是否是一个文件
print(path_instance.is_file()) # True
# Path.is_dir()：判断路径是否是一个目录
# Path.name：返回路径的最后一个部分，即文件名或目录名
print(path_instance.name) # 03_file.txt
# Path.stem：返回路径的最后一个部分的主干，即去掉扩展名后的文件名
print(path_instance.stem) # 03_file
# Path.suffix：返回路径的最后一个部分的扩展名
print(path_instance.suffix) # .txt
# Path.parent：返回路径的父目录
print(path_instance.parent) # /home/silei/WorkSpace_git/Pytorch_learn/python_learn
# Path.joinpath(*other)：将路径与其他路径组合成一个新的路径
new_path = path_instance.parent.joinpath("04_file.txt")
print(new_path) # /home/silei/WorkSpace_git/Pytorch_learn/python_learn/04_file.txt
# Path.iterdir()：返回路径下的所有文件和目录的迭代器
for item in path_instance.parent.iterdir():
    print(item) # /home/silei/WorkSpace_git/Pytorch_learn/python_learn/01_start.py, /home/silei/WorkSpace_git/Pytorch_learn/python_learn/02_function.py, /home/silei/WorkSpace_git/Pytorch_learn/python_learn/03_file.txt


# 5.访问文件夹----------------------------------------------------------------
# 获取当前工作目录
print(os.getcwd()) #这里指的是运行命令行python的目录，但是可以切换，结束后再次运行仍然是当前命令行所在目录
# /home/silei/WorkSpace_git/Pytorch_learn
# 切换工作目录
os.chdir("/home/silei/WorkSpace_git/Pytorch_learn/python_learn")
print(os.getcwd()) # /home/silei/WorkSpace_git/Pytorch_learn/python_learn

# 创建新目录
os.mkdir("new_folder") # 在当前工作目录下创建一个名为new_folder的目录
# 删除目录
os.rmdir("new_folder") # 删除当前工作目录下的new_folder目录
# 修改目录名
os.rename("old_folder", "new_folder") # 将当前工作目录下的old_folder目录重命名为new_folder

# 遍历文件夹
# os.walk(top, topdown=True, onerror=None, followlinks=False)
# 生成一个三元组(root, dirs, files)，分别表示当前遍历到的目录路径、该目录下的子目录列表和当前目录下的文件列表
# 注意：os.walk()函数会递归地遍历指定目录及其所有子目录，以及子目录中的...
# 如果topdown参数为True（默认值），则先遍历当前目录，再遍历子目录；如果topdown参数为False，则先遍历子目录，再遍历当前目录
for root, dirs, files in os.walk("/home/silei/WorkSpace_git/Pytorch_learn/python_learn"):
    print("Root:", root) # 当前目录路径
    print("Dirs:", dirs) # 当前目录下的子目录列表
    print("Files:", files) # 当前目录下的文件列表