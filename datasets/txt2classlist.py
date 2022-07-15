def trans(path):
    with open(path,'r') as f:
        content = f.read()
        cls_list = content.split("\n")[:-1]
    return cls_list