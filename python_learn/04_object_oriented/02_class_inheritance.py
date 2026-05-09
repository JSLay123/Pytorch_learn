# 类的继承-------------------------------------------------------------------------------
# 父类的属性和方法会被子类继承，子类可以重写父类的方法，也可以添加新的属性和方法。

# 初始化方法
# 如果子类不写__init_方法，默认会调用父类的_init_方法
# 如果自己写了__init__方法，就需要使用super()函数来调用父类的__init__方法，否则父类的属性就无法被初始化。
# super()函数可以用来调用父类的方法，特别是在子类重写了父类的方法时，可以通过super()来调用父类的版本。
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)  # 调用父类的构造方法
        self.student_id = student_id


# isintance()函数可以用来检查一个对象是否是某个类的实例，或者是否是某个类的子类的实例。
student = Student("Alice", 20, "S12345")
# 一个子类对象同时也是父类的对象，因为子类继承了父类的属性和方法。
print(isinstance(student, Student))  # 输出：True
print(isinstance(student, Person))   # 输出：True
# 但是父类对象不是子类的对象，因为父类没有子类特有的属性和方法。
person = Person("Bob", 30)
print(isinstance(person, Person))    # 输出：True
print(isinstance(person, Student))   # 输出：False

# issubclass()函数可以用来检查一个类是否是另一个类的子类。
print(issubclass(Student, Person))  # 输出：True
print(issubclass(Person, Student))  # 输出：False


# 方法重写
# 子类可以重写父类的方法，提供自己的实现。
class Parent:
    age = 50
    def my_method(self):
        print("This is the parent method.")

    def print_age(self):
        print(f"The age is {self.age}.")

class Child(Parent):
    age = 20
    def my_method(self):
        print("This is the child method.")

class GrandChild(Parent):
    pass

child = Child()
child.my_method()  # 输出：This is the child method.
# 如果子类不重实现父类的方法，那么调用该方法时会执行父类的版本。


# 定义一个传入父类对象的函数，传入子类对象时会调用子类重写的方法，因为子类对象会覆盖父类的方法。
# 这里代表传入任何是属于Person类型的对象，子类就也属于父类对象
def call_parent_method(parent):
    parent.my_method()
call_parent_method(child)  # 输出：This is the child method.
grandchild = GrandChild()   # 没有重写父类的方法，所以调用父类的方法
call_parent_method(grandchild)  # 输出：This is the parent method.

# 静态属性也会被继承和重写
print(Parent.age)  # 输出：50
print(Child.age)   # 输出：20
# 父类的静态属性被父类方法调用时，子类直接继承该父类方法，调用时会使用子类的静态属性值。
child.print_age()  # 输出：The age is 20.
grandchild.print_age()  # 输出：The age is 50.

# 子类可以通过super()函数来调用父类的方法，即使该方法被子类重写了。
class ChildWithSuper(Parent):
    age = 20
    def my_method(self):
        print("This is the child method.")
        super().my_method()  # 调用父类的方法


# 类的多继承-----------------------------------------------------------------------
# Python支持多继承，一个子类可以同时继承多个父类。

# 多继承语法, class Child(Parent1, Parent2, ...):
class Mother:
    def method_a(self):
        print("This is method A from Mother.")
class Father:
    def method_b(self):
        print("This is method B from Father.")
class Child(Mother, Father):
    pass
child = Child()
child.method_a()  # 输出：This is method A from Mother.
child.method_b()  # 输出：This is method B from Father.

# 多继承可能会导致方法冲突，如果多个父类有同名的方法，Python中子类调用该方法时会按照方法解析顺序（MRO）来决定调用哪个父类的方法。
class ParentA:
    def my_method(self):
        print("This is ParentA's method.")
class ParentB:
    def my_method(self):
        print("This is ParentB's method.")
class Child(ParentA, ParentB):
    pass
child = Child()
child.my_method()  # 输出：This is ParentA's method.，因为ParentA在Child的继承列表中排在ParentB前面。

# 如果多个父类有同名的方法，且我想指定调用哪个父类的方法，可以使用super()函数来指定父类。
class ChildWithSuper(ParentA, ParentB):
    def my_method(self):
        print("This is Child's method.")
        super(ParentB, self).my_method()  # 指定调用ParentA的方法
child_with_super = ChildWithSuper()
child_with_super.my_method()  # 输出：This is Child's method. This is ParentA's method.，因为super(ParentB, self)会跳过ParentB，直接调用ParentA的方法。