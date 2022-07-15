from os import popen


def plot(path):
    ploter = popen('tree {}'.format(path))
    print(ploter.read())


plot(r'./')