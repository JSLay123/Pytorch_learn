# 1.类变量-------------------------------------------------------------------------
# 类本身也是一个对象，是type类型的对象
# 类变量是属于类的变量，所有实例共享同一个类变量
# 类变量在类定义时就被创建，并且在内存中只有一份，所有实例都可以访问和修改类变量的值
# 类变量中占据内存空间的是函数还是变量，取决于类变量的类型，如果类变量是一个函数，那么占据内存空间的是函数对象；如果类变量是一个变量，那么占据内存空间的是变量的值
class MyClass:
    class_variable = 0  # 定义一个类变量
# 获取类的名称, 可以通过类的__name__属性获取
print(MyClass.__name__) # MyClass

# 获取类变量的值
print(MyClass.class_variable) # 0
# getattr()：获取对象的属性值，可以指定默认值，如果属性不存在则返回默认值
print(getattr(MyClass, "class_variable")) # 0
print(getattr(MyClass, "class_variable", 1)) # 1

# 动态添加新的类变量
MyClass.new_class_variable = 100
print(MyClass.new_class_variable) # 100
setattr(MyClass, "another_class_variable", 200)
print(MyClass.another_class_variable) # 200

# 赋值类变量的值
MyClass.class_variable = 10
print(MyClass.class_variable) # 10
# setattr()：设置对象的属性值，如果属性不存在则创建属性
setattr(MyClass, "class_variable", 20)
print(MyClass.class_variable) # 20

# 删除类变量
# del MyClass.class_variable
# print(MyClass.class_variable) # AttributeError: type object 'MyClass' has no attribute 'class_variable'
# delattr()：删除对象的属性，如果属性不存在则抛出AttributeError异常
# delattr(MyClass, "class_variable")
# print(MyClass.class_variable) # AttributeError: type object 'MyClass' has no attribute 'class_variable'

# 类变量相当于静态变量，属于类本身，而不是某个实例，所有实例共享同一个类变量的值，修改类变量的值会影响所有实例访问该类变量的值
class Student:
    school_name = "ABC School"  # 定义一个类变量   
student1 = Student()
student2 = Student()
print(student1.school_name) # ABC School
print(student2.school_name) # ABC School
# 修改类变量的值
Student.school_name = "XYZ School"
print(student1.school_name) # XYZ School
print(student2.school_name) # XYZ School

# 类变量的存储
# 类变量在类定义时就被创建，并且在内存中只有一份，所有实例都可以访问和修改类变量的值
# 类变量存储在类对象的__dict__属性中，类对象的__dict__属性是一个字典，存储了类的属性和方法，包括类变量和类方法
print(MyClass.__dict__) 
# {
#   '__module__': '__main__',
#   'class_variable': 20,
#   '__dict__': <attribute '__dict__' of 'MyClass' objects>, 
#   '__weakref__': <attribute '__weakref__' of 'MyClass' objects>, 
#   '__doc__': None
# }


# 2.实例变量和函数---必须有self-------------------------------------------------------------------
# 类变量属于类本身，而实例变量属于对象
# 类变量可以直接通过类名访问，而实例变量必须通过实例对象访问
# 实例变量是属于对象的变量，每个对象都有自己的实例变量，相当于属性
# 实例变量在该类对象创建时被创建，并且每个对象都有自己的实例变量，修改一个对象的实例变量不会影响其他对象的实例变量

# 定义实例函数
class MyClass:
    def instance_method(self, msg:str):  # 定义一个实例函数，需要传入self参数，self参数表示实例对象本身，可以通过self参数访问实例变量和类变量
        print(f"This is an instance method: {msg}")
# 创建一个实例对象
my_instance = MyClass()
# 调用实例函数
my_instance.instance_method("Hello, World!") # This is an instance method: Hello, World!

# 定义实例变量
class Student:
    def __init__(self, name:str, age:int):  # 定义一个构造函数，__init__函数是一个特殊的函数，在创建对象时会自动调用，可以用来初始化对象的属性
        self.name = name  # 定义一个实例变量name，使用self参数来定义实例变量，实例变量属于对象，每个对象都有自己的实例变量
        self.age = age    # 定义一个实例变量age
    
    def display_info(self):  # 定义一个实例函数，使用self参数来定义实例函数，实例函数属于对象，每个对象都有自己的实例函数
        print(f"Name: {self.name}, Age: {self.age}")
# 创建一个实例对象
student1 = Student("Alice", 20)
student2 = Student("Bob", 22)
student1.display_info()  # Name: Alice, Age: 20
student2.display_info()  # Name: Bob, Age: 22

# 实例化对象添加属性
student1.grade = "A"  # 给student1对象添加一个新的实例变量grade
print(student1.grade)  # A
# 因为grade是student1对象的实例变量，实例变量属于对象，每个对象都有自己的实例变量，所以student2对象没有grade属性，访问student2.grade会抛出AttributeError异常
# print(student2.grade)  # AttributeError: 'Student' object has no attribute 'grade'，student2对象没有grade属性


# 3.私有属性和私有函数---------------------------------------------------------------------------
# 一个或者两个下划线开头
# 私有属性和私有函数是以双/单下划线开头的属性和函数，私有属性和私有函数只能在类的内部访问，不能在类的外部访问，
# 私有属性和私有函数在类的外部访问会抛出AttributeError异常，私有属性和私有函数在类的内部可以通过self参数访问，私有属性和私有函数在类的外部可以通过对象._ClassName__PrivateName的方式访问
class MyClass:
    def __init__(self, name:str):
        self.__name = name  # 定义一个私有属性__name，使用双下划线开头，表示这是一个私有属性，私有属性只能在类的内部访问，不能在类的外部访问
    
    def __private_method(self):  # 定义一个私有函数__private_method，使用双下划线开头，表示这是一个私有函数，私有函数只能在类的内部访问，不能在类的外部访问
        print("This is a private method.")
    
    def public_method(self):  # 定义一个公共函数public_method，可以在类的内部和外部访问
        print(f"The name is: {self.__name}")  # 在公共函数中访问私有属性__name
        self.__private_method()  # 在公共函数中调用私有函数__private_method 

# 如果非要在类的外部访问私有属性和私有函数，可以通过对象._ClassName__PrivateName的方式访问
my_instance = MyClass("Alice")
print(my_instance._MyClass__name)  # Alice，通过对象._ClassName__PrivateName的方式访问私有属性__name
my_instance._MyClass__private_method()  # This is a private method.，通过对象._ClassName__PrivateName的方式访问私有函数__private_method

# 单下划线和双下划线的区别：
# 单下划线开头的属性和函数是一个约定，表示这是一个私有属性和私有函数，虽然可以在类的外部访问，但不建议在类的外部访问，单下划线开头的属性和函数在类的外部访问不会抛出AttributeError异常，但会有一个警告，表示这是一个私有属性和私有函数，不建议在类的外部访问
# 双下划线开头的属性和函数是一个真正的私有属性和私有函数，不能在类的外部访问，双下划线开头的属性和函数在类的外部访问会抛出AttributeError异常，表示这是一个私有属性和私有函数，不能在类的外部访问

# 4.类方法和静态方法---------------------------------------------------------------------------
# 和之前提到的类变量是一个层级的东西，属于类本身，而不是某个实例，所有实例共享同一个类变量的值，修改类变量的值会影响所有实例访问该类变量的值
# 类方法和静态方法是属于类本身的方法，而不是某个实例的方法

# 类方法使用@classmethod装饰器定义，类方法的第一个参数是cls，表示类本身，可以通过cls参数访问类变量和类方法，类方法可以通过类名或者对象调用
# 静态方法使用@staticmethod装饰器定义，静态方法没有默认参数，可以通过类名或者对象调用，静态方法不能访问类变量和实例变量，只能访问静态方法内部定义的变量
class MyClass:
    class_variable = 0  # 定义一个类变量
    
    @classmethod
    def class_method(cls, msg:str): # 默认要有一个cls
        print(cls.__name__) # MyClass，类方法可以通过cls参数访问类的属性和方法
        print(cls.class_variable) # 0，类方法可以通过cls参数访问类变量
        print(f"This is a class method. Class variable: {cls.class_variable}, Message: {msg}")

    # 静态方法没有默认参数，不能访问类变量和实例变量，只能访问静态方法内部定义的变量
    # 理解为就是一个正常的和类外一样的一个函数，只不过写在类内面，和类的关系不大，静态方法不能访问类变量和实例变量，只能访问静态方法内部定义的变量
    @staticmethod
    def static_method(msg:str):
        print(MyClass.__name__) # MyClass，静态方法可以通过类名访问类的属性和方法
        print(f"This is a static method. Message: {msg}")

    def instance_method(self, msg:str):
        self.class_method(msg) # 在实例方法中调用类方法
        self.static_method(msg) # 在实例方法中调用静态方法

# 调用类方法和静态方法
MyClass.class_method("Hello, Class!")  # This is a class method. Class variable: 0，通过类名调用类方法
MyClass.static_method("Hello, Static!")  # This is a static method.，通过类名调用静态方法


# 5.类中常用的特殊方法---------------------------------------------------------------------------
# __init__()：构造函数，在创建对象时自动调用，用于初始化对象的属性

class MyDate:
    def __init__(self, year:int, month:int, day:int):
        self.year = year
        self.month = month
        self.day = day
    
    def __str__(self):  # 该用法主要面向用户，返回一个可读性较好的字符串表示，适合打印输出
        return f"{self.year}-{self.month:02d}-{self.day:02d}"
    
    def __repr__(self): # 该用法主要面向开发者，返回一个官方字符串表示，适合调试和开发使用
        return f"MyDate(year={self.year}, month={self.month}, day={self.day})" 
    
    def __eq__(self, value):
        if isinstance(value, MyDate):
            return self.year == value.year and self.month == value.month and self.day == value.day
        return False
    
    def __hash__(self):
        print("Calculating hash value...")
        return hash((self.year + 101 * self.month + 101 * self.day))
    
    # bool方法用于再对象被bool函数求解的时候返回一个布尔值
    # 如果类没有实现这个方法，就会默认返回True，除非对象的__len__()方法返回0或者对象的__bool__()方法返回False，否则对象在布尔上下文中被认为是True
    def __bool__(self):
        print("Calculating boolean value...")
        return self.year != 0 and self.month != 0 and self.day != 0
    
    # 当对象被销毁时自动调用，用于清理对象的资源，例如关闭文件、释放内存等，通常不需要手动调用该方法，Python会在对象被垃圾回收时自动调用该方法
    # 一般不会使用
    # 和C++的析构函数的区别：
    # C++的析构函数是在对象生命周期结束时自动调用的，用于释放对象占用的资源，例如内存、文件句柄等，C++的析构函数是通过对象的生命周期来管理资源的，C++的析构函数是在对象被销毁时自动调用的
    # 而Python的__del__方法是在对象被垃圾回收时自动，至于回收时间是由Python的垃圾回收机制来决定的，Python的垃圾回收机制是基于引用计数和垃圾回收算法的，
    # 因此Python的__del__方法的调用时间是不确定的，可能会在对象被销毁时调用，也可能会在程序退出时调用，也可能永远不会调用，
    # 因此不建议在Python中使用__del__方法来管理资源，应该使用上下文管理器或者try-finally语句来管理资源。
    def __del__(self):
        print(f"MyDate object with year={self.year}, month={self.month}, day={self.day} is being destroyed.")
        
        
        
# __str__()：字符串表示方法，返回对象的字符串表示，使用print()函数打印对象时会调用该方法
date = MyDate(2024, 6, 1)
print(date) # 2024-06-01，使用print()函数打印对象时会调用__str__()方法，返回对象的字符串表示

# __repr__()：官方字符串表示方法，返回对象的官方字符串表示，使用repr()函数获取对象的字符串表示时会调用该方法
print(repr(date)) # MyDate(year=2024, month=6, day=1)，使用repr()函数获取对象的字符串表示时会调用__repr__()方法

# __eq__()：相等比较方法，定义对象的相等比较规则，使用==运算符比较对象时会调用该方法
date1 = MyDate(2024, 6, 1)
date2 = MyDate(2024, 6, 1)
date3 = MyDate(2024, 6, 2)
print(date1 == date2) # True，使用==运算符比较对象时会调用__eq__()方法，返回对象的相等比较结果
print(date1 == date3) # False，使用==运算符比较对象时会调用__eq__()方法

# __hash__()：哈希方法，定义对象的哈希值，使用hash()函数获取对象的哈希值时会调用该方法
# 哈希值用于在哈希表中存储对象，例如在set和dict中使用对象作为键时会调用该方法获取对象的哈希值
date_set = set()
date_set.add(date1) # 添加对象到set中时会调用__hash__()方法获取对象的哈希值, print出Calculating hash value...，表示正在计算哈希值

# __bool__()：布尔方法，定义对象的布尔值，使用bool()函数获取对象的布尔值时会调用该方法
print(bool(date1)) # True
date_zero = MyDate(0, 0, 0)
print(bool(date_zero)) # False

# __del__()：析构函数，在对象被销毁时自动调用，用于清理对象的资源

# __call__()：可调用对象方法，使对象可以像函数一样被调用，使用对象()的方式调用对象时会调用该方法
# 可调用对象方法使对象可以像函数一样被调用，使用对象()的方式调用对象时会调用该方法，可以在__call__()方法中定义对象被调用时的行为，例如执行某个操作、返回某个值等
class MyCallable:
    def __call__(self, *args, **kwargs):
        print(f"MyCallable object called with args: {args}, kwargs: {kwargs}")

my_callable = MyCallable()
my_callable(1, 2, 3, key="value") # MyCallable object called with args: (1, 2, 3), kwargs: {'key': 'value'}


# 6.property类----------------------------------------------------------------------------
# property类是一个内置的类，用于创建属性，property类可以将一个方法转换为属性，使得可以通过访问属性的方式调用方法
# property类的构造函数接受四个参数，分别是fget、fset、fdel和doc，分别表示属性的获取方法、设置方法、删除方法和文档字符串
# 主要对私有属性进行封装，提供一个公共的接口来访问和修改私有属性，使用property类创建属性后，可以通过访问属性的方式调用方法，使得代码更加简洁和易读，同时也可以对属性的访问进行控制，例如在获取属性值时进行一些验证或者在设置属性值时进行一些限制等
class Student:
    def __init__(self, name:str):
        self._name = name  # 定义一个私有属性_name，使用单下划线开头，表示这是一个私有属性，私有属性只能在类的内部访问，不能在类的外部访问
    
    def get_name(self):  # 定义一个获取方法get_name，用于获取属性_name的值
        return self._name
    
    def set_name(self, name:str):  # 定义一个设置方法set_name，用于设置属性_name的值
        self._name = name
    
    # 构造property类就像创建一个代理一样，property类会将get_name方法和set_name方法转换为一个属性name，使得可以通过访问属性name的方式调用get_name方法和set_name方法
    name = property(fget=get_name, fset=set_name)  # 使用property类创建一个属性name，属性的获取方法是get_name，设置方法是set_name

# 使用property类创建属性后，可以通过访问属性的方式调用方法
student = Student("Alice")
print(student.name) # Alice，通过访问属性name的方式调用get_name方法，获取属性_name的值
student.name = "Bob" # 通过访问属性name的方式调用set_name方法，设置属性_name的值
print(student.name) # Bob，通过访问属性name的方式调用get_name方法，获取属性_name的值


# 7.@property装饰器 和 @property.setter装饰器----------------------------------------------------------------------------
# @property装饰器是一个内置的装饰器，用于创建属性，@property装饰器可以将一个方法转换为属性，使得可以通过访问属性的方式调用方法
# @property装饰器是property类的一个简化语法，可以通过在方法定义前使用@property装饰器来创建属性
class Student:
    def __init__(self, name:str):
        self._name = name  # 定义一个私有属性_name，使用单下划线开头，表示这是一个私有属性，私有属性只能在类的内部访问，不能在类的外部访问
    
    # 相当于类外可以将age当作一个属性去使用
    @property
    def name(self):  # 定义一个获取方法name，用于获取属性_name的值，使用@property装饰器创建一个属性name，使得可以通过访问属性name的方式调用该方法
        return self._name
    
    @name.setter
    def name(self, name:str):  # 定义一个设置方法name，用于设置属性_name的值，使用@setter装饰器创建一个属性name，使得可以通过访问属性name的方式调用该方法
        self._name = name

# 使用@property装饰器创建属性后，可以通过访问属性的方式调用方法
student = Student("Alice")
print(student.name) # Alice，通过访问属性name的方式调用name方法，获取属性_name的值
student.name = "Bob" # 通过访问属性name的方式调用name方法，设置属性_name的值
print(student.name) # Bob，通过访问属性name的方式调用name方法，获取属性_name的值

# 只读property 和 删除property
class Square:
    def __init__(self, side_length:float):
        self.__side_length = side_length  # 定义一个私有属性_side_length，使用单下划线开头，表示这是一个私有属性，私有属性只能在类的内部访问，不能在类的外部访问
        self.__area = None
    
    @property
    def side_length(self):
        return self.__side_length
    
    @side_length.setter
    def side_length(self, side_length:float):
        self.__side_length = side_length
        self.__area = None  # 当边长发生变化时，重置面积的值为None，表示需要重新计算面积

    @property
    def area(self):  # 定义一个获取方法area，用于获取属性_area的值，使用@property装饰器创建一个只读属性area，使得可以通过访问属性area的方式调用该方法
        if self.__area is None:  # 如果_area的值为None，表示还没有计算过面积，需要计算面积
            self.__area = self.__side_length ** 2  # 计算面积，并将结果存储在私有属性__area中，下次访问area属性时直接返回__area的值，不需要重新计算面积
        return self.__area
    
    @area.deleter
    def area(self):  # 定义一个删除方法area，用于删除属性_area的值，使用@deleter装饰器创建一个属性area，使得可以通过访问属性area的方式调用该方法
        self.__area = None  # 删除_area的值，将其重置为None，表示需要重新计算面积

square = Square(4)
print(square.area) # 16，通过访问属性area的方式调用area方法，获取属性_area的值，第一次访问时会计算面积并返回结果
print(square.area) # 16，通过访问属性area的方式调用area方法，获取属性_area的值，第二次访问时直接返回之前计算的结果，不需要重新计算面积
square.side_length = 5 # 通过访问属性side_length的方式调用side_length方法，设置属性_side_length的值为5，同时重置属性_area的值为None，表示需要重新计算面积
print(square.area) # 25，通过访问属性area的方式调用area方法，获取属性_area的值，由于之前重置了_area的值为None，所以需要重新计算面积并返回结果
del square.area # 通过访问属性area的方式调用area方法，删除属性_area的值，将其重置为None，表示需要重新计算面积
print(square.area) # 25，通过访问属性area的方式调用area方法，获取属性_area的值，由于之前删除了_area的值，所以需要重新计算面积并返回结果