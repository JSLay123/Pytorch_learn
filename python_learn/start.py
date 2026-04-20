# 1. 变量的定义------------------------------------------------------------------------------
# 定义一个变量，就是开辟一个内存空间，来存储数据
# 变量的类型由数据的类型决定
name ="Tom"

# 新数据类型实际是新的内存空间，原来的内存空间被垃圾回收机制回收了，也就找不到了
# 最好是一个变量只存储一种数据，避免混乱
name = 10

# 2.String----------------------------------------------------------------------------------
#  单引号、双引号 之间的字符序列
msg = "Hello, 'World'!"
# 转义字符 \ 后面跟随一个特殊字符，表示这个特殊字符的特殊含义
# 比如 \n 表示换行，\t 表示制表符，\\ 表示一个反斜杠， \' 表示一个单引号，\" 表示一个双引号
desc = r'This is \ hello \ '    # r表示原始字符串，不进行转义

# 字符串的连接
greeting = "Hello, " + "World! " + msg
print(greeting)    # Hello, World!Hello, 'World'!

# \当作续行符，表示当前行未结束，下一行继续，自动连续字符串
desc = "aaaaaaa" \
        "sssssss"

# f-string格式化字符串，{}内可以放变量或表达式，代表name是传入的变量
message = f'Hello, {name}!'  

# 获取字符串的长度
length = len(message)
# 获取字符串的某个字符
first_char = message[0]    # H
# 打印变量的类型
print(type(message), type(first_char))     # <class 'str'>

# 3.算术运算---------------------------------------------------------------------------------
# ➗ / 代表除法，结果是一个浮点数
result = 10 / 2   # 5.0

# 整数和浮点数的除法结果是浮点数

# 数中的下划线可以用来分隔数字，使其更易读
count = 100_000_000   # 100000000

# 4.bool------------------------------------------------------------------------------------
# bool类型只有两个值：True和False
# 数值0,空字符串""，空列表[]，空字典{}，空集合set()等都被视为False，其他值都被视为True

# 5.类型转换---------------------------------------------------------------------------------
# int() float() str() bool() list() tuple() dict() set()
# int() 将一个数值或字符串转换为整数
# 原值仍然存在且不变，int()返回一个新的整数对象
num_str = "123"
num_int = int(num_str)   # 123

input_str = input("请输入一个数字: ")  # 从用户输入获取一个字符串
input_num = int(input_str)  # 尝试将输入的字符串转换为整数
print(f"你输入的数字是: {input_num}")

# 6.操作符-----------------------------------------------------------------------------------
# 比较操作符 ： <, >, <=, >=, ==, !=
# 逻辑操作符 ： and, or, not
# 逻辑操作符的优先级：not > and > or，且左右都需要是布尔值


# 7.条件语句---------------------------------------------------------------------------------
# if 条件1:
# if 条件1:  else：
# if 条件1:  elif 条件2:  else：

# 8.循环语句---------------------------------------------------------------------------------
# for 变量 in 可迭代对象:
for i in range(5):
    print(i)    # 0 1 2 3 4
# range(start, stop, step) 生成一个整数序列，默认start=0，step=1

# while 条件: 循环体

# break 退出当前循环， continue 跳过当前循环的剩余部分，进入下一次循环， pass 占位符，表示什么都不做

 # 9.函数------------------------------------------------------------------------------------
# def 函数名(参数列表):
def greet(name : str) -> str:   # 函数参数和返回值的类型提示
    return f"Hello, {name}!"
# 函数调用
message = greet("Alice")

# 函数的缺省参数，如果调用函数时没有提供该参数，则使用默认值
# 缺省值一定在必需参数的后面，否则会导致语法错误
def greet(age : int, name: str = "World") -> str:
    return f"Hello, {name}!, You are {age} years old." 

# 函数的关键字参数，调用函数时可以指定参数的名称，这样就不需要按照定义时的顺序传递参数了
message = greet(age=30, name="Alice")  # Hello, Alice!, You are 30 years old.
message = greet(name="Bob", age=25)    # Hello,

# 10.递归函数--------------------------------------------------------------------------------
# 递归函数是指在函数体内调用自身的函数
# 注意递归函数必须有一个终止条件，否则会导致无限递归，最终导致栈溢出错误
# 不建议使用大递归，因为Python的默认递归深度限制为1000，超过这个限制会抛出RecursionError异常
def factorial(n: int) -> int:
    """计算n的阶乘"""
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)
    
def print_1_to_n(n: int) -> None:
    """打印从1到n的数字"""
    if n > 0:
        print_1_to_n(n - 1)
        print(n)

def summation(n: int) -> int:
    """计算从1到n的和"""
    if n == 0:
        return 0
    else:
        return n + summation(n - 1)

# 11.匿名函数--------------------------------------------------------------------------------
# lambda 参数列表: 表达式
# 匿名函数没有名字，通常用于需要一个函数对象但又不想定义一个正式函数的场合，比如作为参数传递给高阶函数
# lambda表达式中的参数可以有默认值，也可以使用*args和**kwargs来接受任意数量的位置参数和关键字参数
add = lambda x, y: x + y
result = add(3, 5)   # 8

# lambda表达式也可以直接作为参数传递给高阶函数，比如map()、filter()、sorted()等
def apply_operation(x: int, y: int, add) -> int:
    """对x和y应用operation函数"""
    return add(x, y)

# lambda表达式作为函数返回值
def make_multiplier(n: int):
    """返回一个函数，这个函数将输入乘以n"""
    return lambda x: x * n
double = make_multiplier(2)  # 返回一个函数，这个函数将输入乘以2
result = double(5)   # 10   

# 12.list-----------------------------------------------------------------------------------
# list是一个有序的可变容器，可以存储任意类型的元素
numbers = [1, 2, 3, 4, 5]
# list的常用操作：append()、extend()、insert()、remove()、pop()、clear()、index()、count()、sort()、reverse()等
numbers.append(6)   # 在列表末尾添加一个元素
numbers.insert(0, 0)   # 在列表的指定位置插入一个元素
numbers.remove(3)   # 从列表中删除第一个出现的指定元素
last_number = numbers.pop()   # 从列表中删除并返回最后一个元素
pop_num = numbers.pop(0)    # 从列表中删除并返回指定位置的元素
numbers.clear()    # 删除列表中的所有元素
index_of_2 = numbers.index(2)   # 返回列表中第一个出现的指定元素的索引
count_of_2 = numbers.count(2)    # 返回列表中指定元素的出现次数
numbers.sort()    # 对列表进行排序
numbers.reverse() # 反转列表中的元素顺序

# 13.tuple----------------------------------------------------------------------------------
# tuple是一个有序的不可变容器，可以存储任意类型的元素，定义时使用圆括号()，元素之间用逗号分隔
empty_tuple = ()
single_tuple = (1,)   # 定义一个只有一个元素的tuple，必须在元素后面加一个逗号，否则会被当作普通的表达式
point = (1, 2)
default_tuple = (1, "Hello", [1, 2, 3], (4, 5), {"key": "value"})
x, y = point   # tuple的解包，x=1, y=2
# 不加括号的tuple
coordinates = 1, 2, 3   # 这也是一个tuple，等价于 (1, 2, 3)
# 可以直接使用索引访问tuple中的元素
first_point = point[0]   # 1
# 函数的多返回值其实是返回一个tuple
def get_coordinates() -> tuple:
    """返回一个包含x和y坐标的tuple"""
    x = 10
    y = 20
    return x, y   # 返回一个tuple (10, 20)
# tuple的常用操作：count()、index()、len()、max()、min()等
print(point.count(1))   # 1
print(point.index(2))   # 1
print(len(point))   # 2
print(max(point))   # 2
print(min(point))   # 1
# list转换为tuple
numbers_list = [1, 2, 3, 4, 5]
numbers_tuple = tuple(numbers_list)   # (1, 2, 3, 4, 5)
# tuple的不可变性意味着一旦创建了一个tuple，就不能修改它的元素，但如果tuple中包含可变对象，比如列表，那么这个可变对象是可以修改的
my_tuple = (1, [2, 3], 4)
# 元组相加，相乘：变成一个新的元组
tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)
combined_tuple = tuple1 + tuple2   # (1, 2, 3, 4, 5, 6)
repeated_tuple = tuple1 * 2   # (1, 2, 3, 1, 2, 3)，相乘就是复制拼接

# 14.排序----------------------------------------------------------------------------------
# sort()函数与sorted()函数的区别：
# sort()是列表对象的方法，会直接修改原列表，返回值为None；
# sorted()是一个内置函数，会返回一个新的排序后的列表，原列表不变
numbers = [5, 2, 9, 1, 5, 6]
sorted_numbers = sorted(numbers)   # [1, 2, 5, 5, 6, 9]
print(numbers)   # [5, 2, 9, 1, 5, 6]，原列表不变

numbers.sort(reverse=False)   # [1, 2, 5, 5, 6, 9]，reverse=True就是降序排序
print(numbers)   # [1, 2, 5, 5, 6, 9]，原列表被修改了
# sorted()函数还可以接受一个key参数，用于指定一个函数，这个函数会被用来从每个元素中提取一个用于排序的键
words = ["banana", "apple", "cherry", "date"]
sorted_words = sorted(words, key=len)   # ['date', 'apple', 'banana', 'cherry']，按照字符串长度排序
# sort()方法也可以接受一个key参数，使用方法与sorted()函数相同
words.sort(key=len)   # ['date', 'apple', 'banana', 'cherry']，按照字符串长度排序

student = [("Tom", 85), ("Alice", 92), ("Bob", 78)]
# sorted()函数的key参数可以使用lambda表达式来指定一个匿名函数，这个匿名函数接受一个元素作为输入，返回一个用于排序的键，返回的就是需要排序的值
# sorted()函数也可以自己定义一个函数来作为key参数，这个函数接受一个元素作为输入，返回一个用于排序的键
# 按照学生的成绩排序
sorted_students = sorted(student, key=lambda x: x[1], reverse=True)   # [('Alice', 92), ('Tom', 85), ('Bob', 78)]
# 按照学生的名字排序
sorted_students = sorted(student, key=lambda x: x[0])   # [('Alice', 92), ('Bob', 78), ('Tom', 85)]

# 15.unpack list--------------------------------------------------------------------------------
# 列表的解包是指将列表中的元素分别赋值给多个变量
numbers = [1, 2, 3]
a, b, c = numbers   # a=1, b=2, c=3
# 如果变量的数量与列表中的元素数量不匹配，会导致ValueError异常
# a, b = numbers   # ValueError: too many values to unpack (expected 2)
# 使用*运算符来解包列表，可以将剩余的元素收集到一个新的列表中
numbers = [1, 2, 3, 4, 5]
a, b, *rest = numbers   # a=1, b=2, rest=[3, 4, 5]
# 也可以将列表中的元素解包到函数的参数中
def add(x: int, y: int) -> int: