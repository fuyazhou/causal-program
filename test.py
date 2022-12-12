import traceback


def func(num1, num2):
    try:
        x = num1 * num2
        y = num1 / num2
        return x, y
    except:
        traceback.print_exc()


func(1, 0)
