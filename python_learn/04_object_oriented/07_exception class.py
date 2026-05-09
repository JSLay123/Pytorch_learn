# 异常类

# 使用 try...except...finally 语句来处理异常，可以捕获特定类型的异常并进行处理，也可以捕捉父类异常来捕获所有子类异常。
# 或者使用 except Exception 来捕获所有异常。Exception 是所有内置异常的基类

# 所以我们应该先抓 子类型 异常，再抓 父类型 异常，否则父类型异常会捕获所有子类型异常，导致子类型异常的 except 块永远不会被执行。
try:
    x = 1 / 0  # 这将引发 ZeroDivisionError
except ZeroDivisionError as e:
    print("Caught a ZeroDivisionError:", e)
except Exception as e:
    print("Caught a general exception:", e)
finally:
    print("This will always be executed.")

# raise 异常
# 可以使用 raise 语句来主动引发一个异常，通常用于在函数中检测到错误条件时引发异常。
def add(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError("Both a and b must be numbers.")
    return a + b

# 自己定义异常类
# 可以通过继承内置的 Exception 类来定义自己的异常类，这样可以创建更具体的异常类型，便于在代码中区分不同的错误情况。
class MyCustomError(Exception):
    def __init__(self, *args):
        super().__init__(args)  # 调用父类的构造方法，传入参数

try:
    raise MyCustomError("This is a custom error message.")
except MyCustomError as e:
    print("Caught MyCustomError:", e)