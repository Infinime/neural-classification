
############################################################
# Imports
############################################################

import perceptrons_data as data
import math

# Include your imports here, if any are used.



############################################################
# Section 1: Perceptrons
############################################################
def longest(a):
    maxi = ''
    for x in a:
        if len(x)>len(maxi):
            maxi = x
    return maxi

def sign_dot(a,b):
    # compute sign(a.b)
    return dot(a,b)>=0

def dot(a,b):
    # compute (a.b)
    sum =0
    sett = set(a.keys()).intersection(b.keys())
    for x in sett:
        sum += (a[x]*b[x])
    return sum

def add_vectors(a, b, sub=False):
    c = {}
    sett = set(a.keys()).intersection(b.keys())
    for x in sett:
        if sub:
            c[x] = a[x]-b[x]
        else:
            c[x] = a[x]+b[x]
    for x in [x for x in list(a.keys())+list(b.keys()) if x not in sett]:
        if x in a:
            c[x] = a[x]
        else:
            c[x] = b[x]
    return c

def dict_index(dict, value):
    for x in dict:
        if dict[x] == value:
            return x


class BinaryPerceptron(object):
    def __init__(self, examples, iterations):
        self.w = {}
        for n in range(iterations):
            for example in examples:
                x, y = example
                caret_y = sign_dot(self.w, x)
                if caret_y != y:
                    if y:
                        self.w = add_vectors(self.w, x)
                    else:
                        self.w = add_vectors(self.w, x, sub=True)

    def predict(self, x):
        return(sign_dot(self.w, x))


# train = [({"x1":1}, True), ({"x2":1}, True), ({"x1":-1}, False), ({"x2":-1}, False)]
# test = [{"x1":1}, {'x1':1, 'x2': 1}, {'x1':-1, "x2":1.5}, {'x1':-0.5, 'x2':-2}]
# p=BinaryPerceptron(train, 1)
# print([p.predict(x) for x in test])


def argmax_dot(w, x):
    # Accepts a dict of vectors and a test vector and calculates what argmax(wlk.x) is
    vals ={}
    for n in w:
        vals[n]=dot(w[n],x)
    return dict_index(vals, max(list(vals.values())))


class MulticlassPerceptron(object):
    def __init__(self, examples, iterations):
        bigarr = set()
        self.w = {} # array of weight vectors
        for n in range(iterations):
            for s in examples:
                x, y = s
                bigarr2 = set()
                bigarr = bigarr.union(x.keys())
                bigarr2 = bigarr2.union([y])
                # initialize weight
                for key in bigarr2:
                    if key not in self.w:
                        self.w[key] = {}
                        for ke in bigarr:
                            self.w[key][ke] = 0
                caret_y = argmax_dot(self.w, x)
                if y != caret_y:
                    self.w[y] = add_vectors(self.w[y], x)
                    self.w[caret_y] = add_vectors(self.w[caret_y], x, sub=True)

    def predict(self, x):
        return argmax_dot(self.w, x)

# train = [({"x1":1}, 1), ({"x1":1, "x2":1}, 2), ({"x2":1}, 3), ({"x1":-1, "x2":1}, 4), ({"x1":-1}, 5), ({"x1": -1, "x2": -1}, 6), ({"x2": -1}, 7), ({"x1":1, "x2":-1}, 8)]
# p=MulticlassPerceptron(train, 10)
# print([p.predict(x) for x, y in train])

############################################################
# Section 2: Applications
############################################################

class IrisClassifier(object):

    def __init__(self, d):
        for e in range(len(d)):
            # l,w,s,p
            entry=list(d[e])
            entry[0] = dict(zip(("l", 'w', 's', 'p'), entry[0]))
            d[e] = entry
        self.model = MulticlassPerceptron(d, 526)
        #has 98% acc. against training data

    def classify(self, instance):
        return self.model.predict(dict(zip(("l", 'w', 's', 'p'),instance)))


# --------------TEST--------------------
# c = IrisClassifier(data.iris)
# print(c.classify((5.1, 3.5, 1.4, 0.2)))
# print(c.classify((5.4, 3.4, 1.7, 0.2))) # setosa
# print(c.classify((6.7, 3.1, 5.6, 2.4))) # versi
# wrong = 0
# for entry in data.iris:
#     print(dict(zip([c.classify(entry[0].values())], [entry[1]])))
#     if c.classify(entry[0].values())!=entry[1]:
#         wrong+=1
# print(len(data.iris), wrong)


class DigitClassifier(object):

    def __init__(self, d):
        self.data_range = len(longest([entry[0] for entry in d]))
        for e in range(len(d)):
            entry=list(d[e])
            entry[0] = dict(zip((n for n in range(len(entry[0]))), entry[0]))
            d[e] = entry
        self.model = MulticlassPerceptron(d, 11)
        #has 97.3% acc. against training data without too many passes 12

    def classify(self, instance):
        return self.model.predict(dict(zip((n for n in range(len(instance))),instance)))


# --------------TEST--------------------
# c = DigitClassifier(data.digits)
# wrong = 0
# for entry in data.digits:
#     if c.classify(entry[0].values())!=entry[1]:
#         wrong+=1
# print(len(data.digits), wrong, wrong/len(data.digits)*100)


class BiasClassifier(object):

    def __init__(self, d):
        for e in range(len(d)):
            pums = {}
            d[e] = list(d[e])
            if d[e][0]>1:
                pums["t"] = d[e][0]
                pums['b'] = 1
            else:
                pums["t"] = d[e][0]
                pums['b'] = 0
            d[e][0] = pums
        self.p_bias = BinaryPerceptron(d, 100)
        a=1

    def classify(self, instance):
        gt1 = int(instance>1)
        token = abs(instance) 
        token = token if token!=0 else 0.23
        # if gt1: token = instance-1
        self.instance = {"t":token, 'b':gt1}
        return self.p_bias.predict(self.instance)


# # --------------TEST--------------------
# c = BiasClassifier(data.bias)
# wrong = 0
# for x in (-1, 0, 0.5, 1.5, 2):
#     print(c.classify(x))


class MysteryClassifier1(object):

    def __init__(self, d):
        self.dataset=[]
        for e in range(len(d)):
            pums = {}
            d[e] = list(d[e])
            x, y = d[e][0][0], d[e][0][1]
            pums['r'] = int(x**2+y**2)
            pums[1] = 1
            self.dataset += [(pums, d[e][1])]
        self.p_bias = BinaryPerceptron(self.dataset, 10)
        # ~100%(!!!) accuracy vs source data, speedy too, wow.
        a=1

    def classify(self, d):
        pums = {}
        x, y = d
        if x == y == 0:
            x, y = [0.01,0.01]
        pums['r'] = int(x**2 + y**2)
        pums[1] = 1
        return self.p_bias.predict(pums)

# --------------TEST--------------------
c = MysteryClassifier1(data.mystery1)
w = 0
for x in (data.mystery1):
    if c.classify((x[0][0], x[0][1])) != x[1]:
        w+=1
print('error:',w/len(data.mystery1)*100)
breakpoint()

print([c.classify(x) for x in ((0,0), (0,1), (-1,0), (1,2), (-3,-4))])

class MysteryClassifier2(object):

    def __init__(self, d):
        for e in range(len(d)):
            pums = {}
            d[e] = list(d[e])
            pums['x'], pums['y'], pums['z'] = d[e][0][0], d[e][0][1], d[e][0][2]
            pums['a'] = int(pums['x']>=0)
            pums['b'] = int(pums['y']>=0)
            pums['c'] = int(pums['z']>=0)
            pums['r'] = int(pums['x']*pums['y']*pums['z']>=0)
            # pums['x'], pums['y'], pums['z'] = abs(pums['x']), abs(pums['y']), abs(pums['z'])
            d[e][0] = pums
        self.p_bias = BinaryPerceptron(d, 100)
        # 100%(!!!) accuracy vs source data
        a=1

    def classify(self, d):
        pums = {}
        pums['x'], pums['y'], pums['z'] = d
        if pums['x'] == pums['y'] == pums['z'] == 0:
            pums['x'], pums['y'], pums['z'] = [0.01,0.01,0.01]
        pums['a'] = int(pums['x']>=0)
        pums['b'] = int(pums['y']>=0)
        pums['c'] = int(pums['z']>=0)
        pums['r'] = int(pums['x']*pums['y']*pums['z']>=0)
        return self.p_bias.predict(pums)

# # --------------TEST--------------------
# c = MysteryClassifier2(data.mystery2)
# w = 0
# for x in (data.mystery2):
#     if c.classify((x[0]['x'], x[0]['y'], x[0]['z'])) != x[1]:
#         w+=1
# print('error:',w/len(data.mystery2)*100)
# breakpoint()

# print([c.classify(x) for x in ((1,1,1), (-1,-1,-1), (1,2,-3), (-1,2,3))])
############################################################
# Section 3: Feedback
############################################################

feedback_question_1 = 0

feedback_question_2 = """
Type your response here.
Your response may span multiple lines.
Do not include these instructions in your response.
"""

feedback_question_3 = """
Type your response here.
Your response may span multiple lines.
Do not include these instructions in your response.
"""
