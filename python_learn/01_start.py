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
# *n代表一个新的小列表
numbers = [1, 2, 3, 4, 5]
a, b, *rest = numbers   # a=1, b=2, rest=[3, 4, 5]

# 16.list的迭代-----------------------------------------------------------------------------------
# 迭代就是遍历
numbers = [1, 2, 3, 4, 5]
for indx, num in enumerate(numbers):   # enumerate()函数返回一个迭代器，生成一个包含索引和值的元组
    print(indx, num)   # 0 1, 1 2, 2 3, 3 4, 4 5
# enumerate()函数还可以接受一个start参数，用于指定索引的起始值，默认是0
for indx, num in enumerate(numbers, start=1):
    print(indx, num)   # 1 1, 2 2, 3 3, 4 4, 5 5

# 迭代器iterator是一个实现了迭代协议的对象，迭代协议包括__iter__()方法和__next__()方法，迭代器可以被用来迭代一个可迭代对象，比如列表、元组、字符串等
# 迭代器的常用函数：iter()、next()等
numbers = [1, 2, 3, 4, 5]
iterator = iter(numbers)   # 创建一个迭代器对象
print(next(iterator))   # 1
print(next(iterator))   # 2
print(next(iterator))   # 3
print(next(iterator))   # 4
print(next(iterator))   # 5
# 当迭代器没有更多的元素可供迭代时，next()函数会抛出StopIteration异常，表示迭代结束了
try:
    print(next(iterator))   # StopIteration异常
except StopIteration:       
    print("迭代结束了")

# 用for循环访问迭代器，for循环会自动调用iter()函数来获取一个迭代器对象，并在每次迭代时调用next()函数来获取下一个元素，直到迭代器没有更多的元素可供迭代时，for循环会自动捕获StopIteration异常并结束循环
numbers = [1, 2, 3, 4, 5]
iterator = iter(numbers)
for i in iterator:
    print(i)    # 1 2 3 4 5

# 17.map()函数------------------------------------------------------------------------------------
# map()函数通过遍历可迭代对象中的每个元素，并将其作为参数传递给指定的函数来生成一个新的迭代器
# iterator = map(function, list)
numbers = [1, 2, 3, 4, 5]
squared_numbers = map(lambda x: x ** 2, numbers)   # <map object at 0x7f8c8c8c8c8c>
# 对迭代器直接list()函数来转换成列表
squared_numbers_list = list(squared_numbers)   # [1, 4, 9, 16, 25]
# map()函数也可以接受多个可迭代对象作为参数，这些可迭代对象的元素会被并行传递给指定的函数
numbers1 = [1, 2, 3]
numbers2 = [4, 5, 6]
summed_numbers = map(lambda x, y: x + y, numbers1, numbers2)   # <map object at 0x7f8c8c8c8c8c>
summed_numbers_list = list(summed_numbers)   # [5, 7, 9]

# 18.filter()函数------------------------------------------------------------------------------------
# filter()函数用来过滤可迭代对象中的元素，留下符合要求的元素，返回一个新的迭代器，这个迭代器包含了所有使指定函数返回True的元素
# iterator = filter(function, list)，不改变原列表，返回一个新的迭代器
numbers = [1, 2, 3, 4, 5]
even_numbers = filter(lambda x: x % 2 == 0, numbers)   # <filter object at 0x7f8c8c8c8c8c>
even_numbers_list = list(even_numbers)   # [2, 4]
# filter()函数也可以接受多个可迭代对象作为参数，这些可迭代对象的元素会被并行传递给指定的函数，只有当指定的函数对所有传递的元素都返回True时，这些元素才会被包含在返回的迭代器中
numbers1 = [1, 2, 3]
numbers2 = [4, 5, 6]
filtered_numbers = filter(lambda x, y: x % 2 == 0 and y % 2 == 0, numbers1, numbers2)   # <filter object at 0x7f8c8c8c8c8c>
filtered_numbers_list = list(filtered_numbers)   # [2]

# 19.reduce()函数------------------------------------------------------------------------------------
# reduce()函数用来对一个可迭代对象中的元素进行累积操作，返回一个单一的结果，这个结果是通过对可迭代对象中的元素依次应用指定的函数来计算得到的
# reduce()函数需要从functools模块中导入
from functools import reduce
numbers = [1, 2, 3, 4, 5]
product = reduce(lambda x, y: x * y, numbers)   # 120，相当于 (((1 * 2) * 3) * 4) * 5
# reduce()函数也可以接受一个可选的初始值作为第三个参数，如果提供了初始值，那么这个初始值会被用作第一次调用指定函数时的第一个参数，而可迭代对象中的第一个元素会被用作第二个参数
product_with_initial = reduce(lambda x, y: x * y, numbers, 10)   # 1200，初始值10乘以列表中的所有元素   

# 20.列表解析--------------------------------------------------------------------------------------
# 列表解析是对一个列表进行操作来生成新列表的过程
# 列表解析的语法：new_list = [expression for item in iterable if condition]
# expression是一个表达式，item是一个变量，iterable是一个可迭代对象，condition是一个可选的条件表达式
numbers = [1, 2, 3, 4, 5]
squared_numbers = [x ** 2 for x in numbers]   # [1, 4, 9, 16, 25]
# 列表解析也可以包含一个条件表达式，用于过滤元素，只有当条件表达式返回True时，元素才会被包含在生成的新列表中
even_squared_numbers = [x ** 2 for x in numbers if x % 2 == 0]   # [4, 16]
# 列表解析还可以包含多个for子句和if子句，用于生成更复杂的列表
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]   # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 21.字典 dict-------------------------------------------------------------------------------------
# dict是一个无序的可变容器，可以存储任意类型的键值对，定义时使用花括号{}，键值对之间用逗号分隔，键和值之间用冒号:分隔
person = {"name": "Alice", "age": 30, "city": "New York"}
# dict的常用操作：
# 访问字典中的值：通过键来访问对应的值，如果键不存在会抛出KeyError异常，可以使用get()方法来避免这个问题，get()方法会返回一个默认值，如果键不存在的话
name = person["name"]   # "Alice"
age = person.get("age", 0)   # 30，如果键"age"不存在，则返回默认值0
# 添加或修改字典中的键值对：直接通过键来赋值，如果键不存在则添加一个新的键值对，如果键已经存在则修改对应的值
person["email"] = "alice@example.com"   # 添加一个新的键值对
person["age"] = 31   # 修改已经存在的键值对
# 删除字典中的键值对：使用del语句或者pop()方法，del语句会直接删除指定的键值对，如果键不存在会抛出KeyError异常
# pop()方法会删除指定的键值对并返回对应的值，如果键不存在会抛出KeyError异常，可以提供一个默认值来避免这个问题
del person["city"]   # 删除键值对"city": "New York"
email = person.pop("email", "No email")   # 删除键值对"email": "alice@example.com"，并返回对应的值。不存在则返回默认值"No email"

# 22.字典解析---------------------------------------------------------------------------------------
# 字典解析是对一个字典进行操作来生成新字典的过程
# new_dict={key:value for key, value in iterable if condition}
numbers = [1, 2, 3, 4, 5]
squared_dict = {x: x ** 2 for x in numbers}   # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
# 字典解析也可以包含一个条件表达式，用于过滤元素，只有当条件表达式返回True时，元素才会被包含在生成的新字典中
even_squared_dict = {x: x ** 2 for x in numbers if x % 2 == 0}   # {2: 4, 4: 16}
# 字典解析还可以包含多个for子句和if子句，用于生成更复杂的字典
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened_dict = {num: num ** 2 for row in matrix for num in row}  

# 字典的合并
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}
# 使用update()方法来合并字典，dict1会被修改，dict2不变
dict1.update(dict2)   # dict1变成{"a": 1, "b": 3, "c": 4}
# 使用字典解包来合并字典，生成一个新的字典，原来的字典不变
# 如果有相同的键，后面的字典会覆盖前面的字典中的键值对
merged_dict = {**dict1, **dict2}   # merged_dict是{"a": 1, "b": 3, "c": 4}，dict1和dict2不变

# 23.set集合----------------------------------------------------------------------------------------
# set是一个无序的可变容器，可以存储任意类型的唯一元素，定义时使用花括号{}，元素之间用逗号分隔
fruits = {"apple", "banana", "cherry"}
# 定义一个空set集合，必须使用set()函数来创建，因为{}默认是一个空字典
empty_set = set()
# set的常用操作：
# 判断元素是否在集合中：使用in运算符来判断一个元素是否在集合中，返回True或False
print("apple" in fruits)   # True
print("orange" in fruits)  # False
# 添加元素到集合中：使用add()方法来添加一个元素到集合中，如果元素已经存在于集合中，则不会有任何效果
fruits.add("orange")   # 添加一个新的元素"orange"到集合中
fruits.add("apple")    # "apple"已经存在于集合中，所以不会有任何效果
# 从集合中删除元素：使用remove()方法来删除一个元素，如果元素不存在于集合中会抛出KeyError异常
# 可以使用discard()方法来避免这个问题，discard()方法会删除一个元素，如果元素不存在于集合中则不会有任何效果
fruits.remove("banana")   # 删除元素"banana"从集合中
fruits.discard("grape")   # "grape"不存在于集合中，所以不会有任何效果

# frozenset是一个不可变的set集合，定义时使用frozenset()函数来创建，传入一个可迭代对象作为参数，例如：list、tuple、set等
immutable_fruits = frozenset(["apple", "banana", "cherry"])
# frozenset对象不支持add()、remove()等修改集合的方法，但支持set的其他方法和操作，比如union()、intersection()、difference()等


# 24.set的解析、运算----------------------------------------------------------------------------------------
# set解析是对一个可迭代对象进行操作来生成新set集合的过程
# new_set = {expression for item in iterable if condition}
numbers = [1, 2, 3, 4, 5]
squared_set = {x ** 2 for x in numbers}   # {1, 4, 9, 16, 25}
# set解析也可以包含一个条件表达式，用于过滤元素，只有当条件表达式返回True时，元素才会被包含在生成的新set集合中
even_squared_set = {x ** 2 for x in numbers if x % 2 == 0}   # {4, 16}
# set解析还可以包含多个for子句和if子句，用于生成更复杂的set集合
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened_set = {num for row in matrix for num in row}   # {1, 2, 3, 4, 5, 6, 7, 8, 9}

# 25.set的集合运算-------------------------------------------------------------------------------------------
# 使用union()方法来计算两个集合的并集，原集合不变，返回一个新的集合，包含了两个集合中的所有元素，重复的元素只会出现一次
set1 = {1, 2, 3}
set2 = {3, 4, 5}
union_set = set1.union(set2)   # {1, 2, 3, 4, 5}
# union()可以适用于list、tuple等可迭代对象，返回一个新的集合
union_set = set1.union([4, 5, 6])   # {1, 2, 3, 4, 5, 6}
# | 运算符也可以用来计算两个集合的并集，但是要求两个操作数必须都是集合类型，不能是其他可迭代对象，否则会抛出TypeError异常
union_set = set1 | set2   # {1, 2, 3, 4, 5}

# 使用intersection()方法来计算两个集合的交集，原集合不变，返回一个新的集合，包含了两个集合中都存在的元素
intersection_set = set1.intersection(set2)   # {3}
# & 运算符也可以用来计算两个集合的交集，但是要求两个操作数必须都是集合类型，不能是其他可迭代对象，否则会抛出TypeError异常
intersection_set = set1 & set2   # {3}

# 使用difference()方法来计算两个集合的差集，原集合不变，返回一个新的集合，包含了在第一个集合中但不在第二个集合中的元素
difference_set = set1.difference(set2)   # {1, 2}
# - 运算符也可以用来计算两个集合的差集，但是要求两个操作数必须都是集合类型，不能是其他可迭代对象，否则会抛出TypeError异常
difference_set = set1 - set2   # {1, 2}

# 使用symmetric_difference()方法来计算两个集合的对称差集，原集合不变，返回一个新的集合，包含了在第一个集合中但不在第二个集合中，或者在第二个集合中但不在第一个集合中的元素
symmetric_difference_set = set1.symmetric_difference(set2)   # {1, 2, 4, 5}
# ^ 运算符也可以用来计算两个集合的对称差集，但是要求两个操作数必须都是集合类型，不能是其他可迭t对象，否则会抛出TypeError异常
symmetric_difference_set = set1 ^ set2   # {1, 2, 4, 5}

# 子集判断：issubset()方法用来判断一个集合是否是另一个集合的子集, issuperset()方法用来判断一个集合是否是另一个集合的超集
is_subset = set1.issubset(set2)   # False
is_superset = set1.issuperset(set2)   # False 

# 26.异常处理---------------------------------------------------------------------------------------------
# （1）什么是异常：异常是指在程序运行过程中发生的错误情况，异常会导致程序的正常执行被中断，并抛出一个异常对象，包含了错误的类型和相关的信息
# （2）异常的类型：Python内置了很多异常类型，比如ValueError、TypeError、KeyError、IndexError、ZeroDivisionError等，每种异常类型都表示一种特定的错误情况
# （3）异常的处理：使用try-except语句来处理异常，try块中的代码是可能会引发异常的代码， 如果在try块中发生了异常，程序会立即跳转到except块来处理这个异常，
#                   except块中的代码会根据异常的类型来执行相应的处理逻辑，如果没有匹配到的except块，异常会继续向上传递，直到被捕获或者导致程序崩溃
try:
    num = int(input("请输入一个数字: "))
    result = 10 / num
    print(f"结果是: {result}")
except ValueError as e:
    print("输入的不是一个有效的数字，请重新输入。" + 'error message: ' + str(e))
except ZeroDivisionError as e:
    print("除数不能为零，请重新输入。" + 'error message: ' + str(e)) 
else:
    print("没有发生异常，程序正常执行。")
finally:
    print("无论是否发生异常，都会执行的代码。")