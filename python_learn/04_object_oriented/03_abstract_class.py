# 抽象类

# 抽象类是一种不能被实例化的类，通常用来定义接口或者提供一些通用的功能。
# 抽象类可以包含抽象方法，这些方法在抽象类中没有具体的实现，必须在子类中重写。
# 抽象类不能来实例化、构造对象，抽象类的作用是为了被继承，子类必须实现抽象类中的抽象方法，否则子类也会成为抽象类。

# 在Python中，可以使用abc模块来定义抽象类和抽象方法。abc模块提供了ABCMeta元类和abstractmethod装饰器来实现抽象类的功能。
from abc import ABC, abstractmethod
class Action(ABC):
    @abstractmethod
    def do_something(self):
        pass  # 抽象方法没有具体的实现

# 定义一个抽象类Action，包含一个抽象方法do_something()。这个方法没有具体的实现，子类必须重写这个方法。
# 如果一个类继承了抽象类但没有实现所有的抽象方法，那么这个类也会成为抽象类，不能被实例化。
class ConcreteAction(Action):
    def do_something(self):
        print("Doing something in ConcreteAction.")

class AnotherAction(Action):
    def do_something(self):
        print("Doing something in AnotherAction.")

def perform_action(action: Action):
    action.do_something()

# 创建ConcreteAction和AnotherAction的实例，并调用perform_action函数来执行它们的do_something方法。
concrete_action = ConcreteAction()
another_action = AnotherAction()
perform_action(concrete_action)  # 输出：Doing something in ConcreteAction.
perform_action(another_action)   # 输出：Doing something in AnotherAction.

# 抽象类可以当作先实现一个规范接口，然后让不同的子类来实现这个接口的具体功能，这样就实现了代码的解耦和灵活性。
# 注意要使用 @abstractmethod装饰器 来标记抽象方法，然后pass来表示该方法没有具体的实现。子类必须重写这个方法，否则子类也会成为抽象类，不能被实例化。