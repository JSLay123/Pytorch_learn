# __new__方法

# 函数的动态参数列表
# *args表示位置参数的动态列表，**kwargs表示关键字参数的动态字典。
# 就是说"*args"会把传入的所有位置参数收集到一个元组中，而"**kwargs"会把传入的所有关键字参数收集到一个字典中。
# 这使得函数可以接受任意数量的参数，提供了很大的灵活性。
def execute(func, *args, **kwargs):
    return func(*args, **kwargs)

execute(print, "Hello", "World", sep="-")  # 输出：Hello-World

# *args 接受了两个位置参数 "Hello" 和 "World"，并将它们收集到一个元组中。
# **kwargs 接受了一个关键字参数 sep="-"，并将它收集到一个字典中。
# 在函数内部，func(*args, **kwargs) 会将这些参数解包并传递给 func 函数，这里就是 print 函数，最终输出了 "Hello-World"。


# __new__方法的原型定义在object类中，所有的类都继承自object，因此所有的类都可以重写__new__方法。
# __new__方法的作用是创建一个对象并返回该对象，__init__方法的作用是初始化对象，__new__方法在__init__方法之前被调用。
# __new__方法是一个静态方法，接受的参数与__init__方法相同，但第一个参数是cls，表示要创建的类。

# 创建一个类对象的过程：
# person = Person("Alice", 30)
# 1.persn = object.__new__(Person，"Alice", 30)  调用object类的__new__方法，创建一个Person类的实例对象person
# 2.person.__init__("Alice", 30) 调用Person类的__init__方法，初始化该对象

# 重写__new__方法
# 通常情况下，如果想要重写new方法，就不用再定义__init__方法了，可以写在一起
class Square(int):
    def __new__(cls, value):
        return super().__new__(cls, value**2)  # 不用再专门写 object.__new__(cls)，直接调用super()就可以了，父类自动就继承了object

square = Square(4)
print(square)  # 输出：16
print(type(square))  # 输出：<class '__main__.Square'>，说明square是Square类的实例对象，而不是int类的实例对象
print(isinstance(square, Square))  # 输出：True
print(isinstance(square, int))  # 输出：True，说明Square类是int类的子类

# 为啥不需要再定义__init__方法了呢？因为__new__方法已经返回了一个对象，这个对象已经被初始化了，所以不需要再调用__init__方法了。
class Student:
    def __new__(cls, name, age):
        instance = super().__new__(cls)  # 创建一个Student类的实例对象
        instance.name = name  # 给该对象添加name属性
        instance.age = age  # 给该对象添加age属性
        return instance  # 返回该对象
