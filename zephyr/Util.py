

class commonReducer(dict):

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.addcounter = 0
        self.iaddcounter = 0
        self.interactcounter = 0
        self.callcounter = 0

    def __add__(self, other):
        result = commonReducer(self)
        for key in other.keys():
            if key in result:
                result[key] = self[key] + other[key]
            else:
                result[key] = other[key]

        self.addcounter += 1
        self.interactcounter += 1

        return result

    def __iadd__(self, other):
        for key in other.keys():
            if key in self:
                self[key] += other[key]
            else:
                self[key] = other[key]

        self.iaddcounter += 1
        self.interactcounter += 1

        return self

    def copy(self):

        return commonReducer(self)

    def __call__(self, key, result):
        if key in self:
            self[key] += result
        else:
            self[key] = result

        self.callcounter += 1
        self.interactcounter += 1