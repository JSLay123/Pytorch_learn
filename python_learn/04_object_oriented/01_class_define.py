# 1.类变量-------------------------------------------------------------------------
# 类变量是属于类的变量，所有实例共享同一个类变量
# 类变量在类定义时就被创建，并且在内存中只有一份，所有实例都可以访问和修改类变量的值
# 类实例化为对象后，对象也占据内存空间，但对象的内存空间占据的是对象的实例属性，而不是类变量，类变量单独占据内存空间
# 类变量中占据内存空间的是函数还是变量，取决于类变量的类型，如果类变量是一个函数，那么占据内存空间的是函数对象；如果类变量是一个变量，那么占据内存空间的是变量的值
class MyClass:
    class_variable = 0  # 定义一个类变量
print(MyClass.__name__) # MyClass

# 获取类变量的值
print(MyClass.class_variable) # 0
# getattr()：获取对象的属性值，可以指定默认值，如果属性不存在则返回默认值
print(getattr(MyClass, "class_variable")) # 0
print(getattr(MyClass, "class_variable", 1)) # 1

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

# 2.实例变量和函数----------------------------------------------------------------------
# 实例变量是属于对象的变量，每个对象都有自己的实例变量，实例变量