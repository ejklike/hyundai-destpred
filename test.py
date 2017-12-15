class X:
    def __init__(self, *args, **kwargs):
        a = args[0]
        c = kwargs.pop('c')
        d = kwargs.pop('d')
        self.a = a
        # self.b = b
        self.c = c
        self.d = d
        print(args)
        print(kwargs)
    
    def y(self):
        return self.a, self.c, self.d

ab = [1, 2]
cd = {
    'c':3,
    'd':4
}

x = X(*ab, **cd)

print(x.y())