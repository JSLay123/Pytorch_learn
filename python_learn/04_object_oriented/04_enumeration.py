# 枚举

# 为什么需要枚举？
# 在编程中，我们经常需要定义一些固定的常量，比如星期几、颜色、状态等。使用枚举可以让代码更清晰、更易维护，同时也可以避免使用魔法数字（magic numbers）或字符串来表示这些常量，从而减少错误。

# 枚举是一种特殊的类，用于定义一组命名的常量。在Python中，可以使用enum模块来创建枚举。

# 1.定义枚举-------------------------------------------------------------------------------
from enum import Enum
# 定义一个枚举类
class Gender(Enum):
    MALE = 1
    FEMALE = 2

# type()函数可以用来检查一个对象是否是某个枚举类的成员
print(type(Gender.MALE))  # 输出: <enum 'Gender'>，并非int类型
print(isinstance(Gender.MALE, Gender))  # 输出: True

# 2.类中的应用-----------------------------------------------------------------------------
class Student:
    def __init__(self, gender: Gender):
        self.gender = gender

student1 = Student(Gender.MALE)
student2 = Student(Gender.FEMALE)
print(student1.gender)  # 输出: Gender.MALE
print(student2.gender)  # 输出: Gender.FEMALE
# 使用枚举
# 访问枚举成员
print(Gender.MALE)  # 输出: Gender.MALE
print(Gender.FEMALE)  # 输出: Gender.FEMALE
# 获取枚举成员的值
print(Gender.MALE.value)  # 输出: 1
print(Gender.FEMALE.value)  # 输出: 2
# 获取枚举成员的名称，就是定义时的变量名
print(Gender.MALE.name)  # 输出: 'MALE'
print(Gender.FEMALE.name)  # 输出: 'FEMALE'

# 3.枚举的访问--------------------------------------------------------------------------------
# 枚举名称和值都是唯一的，不能重复。枚举成员可以通过名称或值来访问。
# 名称用[]访问，值用()访问。
# 通过名称访问枚举成员
print(Gender['MALE'])  # 输出: Gender.MALE
print(Gender['FEMALE'])  # 输出: Gender.FEMALE
# 通过值访问枚举成员
print(Gender(1))  # 输出: Gender.MALE
print(Gender(2))  # 输出: Gender.FEMALE

# 因此可以通过获取到的字符串来访问枚举类型的成员
str_gender = 'MALE'
student1.gender = Gender[str_gender]    # 相当于 student1.gender = Gender.MALE
print(student1.gender)  # 输出: Gender.MALE


# 4.遍历枚举成员--------------------------------------------------------------------------------
# 枚举类型本身是可迭代的，可以使用for循环来遍历枚举成员。
for gender in Gender:
    print(gender)  # 输出: Gender.MALE、Gender.FEMALE
    print(gender.name)  # 输出: 'MALE'、'FEMALE'
    print(gender.value)  # 输出: 1、2


# 5.枚举的继承-----------------------------------------------------------------------------------
# 枚举类可以继承自其他枚举类，但不能继承自普通类。
# 枚举只能继承没有成员的枚举类，不能继承有成员的枚举类。
# 一般不会使用


# 6.枚举的别名和@enum.unique装饰器-----------------------------------------------------------------
# 枚举成员的值可以重复，这时会创建别名。(使用不是很广泛，但也会有一些场景需要使用别名，比如状态码等)
class Status(Enum):
    SUCCESS = 1
    FAILURE = 2
    OK = 1  # OK是SUCCESS的别名

# 默认后创建的成员会成为前面成员的别名，所以Status.OK是Status.SUCCESS的别名。
print(Status.SUCCESS)  # 输出: Status.SUCCESS
print(Status.OK)       # 输出: Status.SUCCESS，因为OK是SUCCESS的别名
for i in Status:
    print(i)  # 输出: Status.SUCCESS、Status.FAILURE, 但不会输出Status.OK，因为它是Status.SUCCESS的别名

print(Status.SUCCESS is Status.OK)  # 输出: True，因为它们是同一个枚举成员
print(Status.SUCCESS == Status.OK)  # 输出: True，因为它们的值相等

# 使用@enum.unique装饰器可以禁止枚举成员的值重复。
from enum import unique
@unique
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
    # YELLOW = 1  # 如果取消注释，会抛出ValueError: duplicate values found in <enum 'Color'>: YELLOW -> RED
# 在这个例子中，@unique装饰器禁止了枚举成员的值重复，如果取消注释YELLOW成员，就会抛出ValueError异常，因为它的值与RED成员重复了。


# 7.定制和拓展--------------------------------------------------------------------------------
# 枚举类可以定制和拓展，可以添加方法、属性等。

# 定制__str_函数来改变枚举成员的字符串表示形式。
class Status(Enum):
    SUCCESS = 1
    FAIL = 2
    def __str__(self):
        return f"Status: {self.name}({self.value})"

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other
        if isinstance(other, str):
            return self.name == other.upper()
        if isinstance(other, Status):
            return self is other
        return False

    
print(Status.SUCCESS)  # 输出: Status: SUCCESS(1)， 如果不定制__str__函数，输出将是Status.SUCCESS
print(Status.FAIL)     # 输出: Status: FAIL(2)

# 定制__eq__函数来改变枚举成员的比较行为。
print(Status.SUCCESS == 1)  # 输出: True，因为定制了__eq__函数，使得枚举成员可以与整数进行比较
print(Status.SUCCESS == 'SUCCESS')  # 输出: True，因为定制了__eq__函数，使得枚举成员可以与字符串进行比较
print(Status.SUCCESS == Status.SUCCESS)  # 输出: True，因为定制了__eq__函数，使得枚举成员可以与其他枚举成员进行比较
print(Status.SUCCESS == Status.FAIL)  # 输出: False，因为定制了__eq__函数，使得枚举成员只能与相同的枚举成员进行比较


# 定制大小比较函数来改变枚举成员的大小比较行为。@total_ordering装饰器可以自动生成其他的比较方法。
from functools import total_ordering
@total_ordering
class Level(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    def __lt__(self, other):
        if isinstance(other, Level):
            return self.value < other.value
        if isinstance(other, int):
            return self.value < other
        if isinstance(other, str):
            return self.name < other.upper() 
        return False
    

# 8.auto()函数--------------------------------------------------------------------------------
# enum模块提供了一个auto()函数，可以自动为枚举成员赋值，避免了手动指定值的麻烦。
# 但是不太建议，还是自己指定值比较好
from enum import auto
class Weekday(Enum):
    MONDAY = auto()
    TUESDAY = auto()
    WEDNESDAY = auto()
    THURSDAY = auto()
    FRIDAY = auto()
    SATURDAY = auto()
    SUNDAY = auto()
# 使用auto()函数后，枚举成员会自动从1开始依次递增赋值，所以MONDAY的值是1，TUESDAY的值是2，以此类推。
print(Weekday.MONDAY.value)  # 输出: 1
print(Weekday.TUESDAY.value)  # 输出: 2
print(Weekday.WEDNESDAY.value)  # 输出: 3