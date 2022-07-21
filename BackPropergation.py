###This was attempting to understand back properagation through derivaives in a n-n network

import random
import math

learningRate = 0.1
weight = random.random()
bais = random.random()

def forawrdP(inp):
    return activation(weight, bais, inp)

def calculateError(pred, expected):
    return (pred - expected)**2

def divCalculateError(pred, expected):
    return 2*(pred - expected)

def activation(weight, bais, inp):
    return 1/(1+math.exp(-(weight*inp + bais)))

def divActivation(weight, bais, inp):
    top = inp*math.exp(-(weight*inp + bais))
    bot = (math.exp(-(weight*inp + bais))+1)**2
    return top/bot



 
def backProp(inp, exp):
    global weight
    #calculateError(forawrdP(inp), exp)
    
    DcDa = divCalculateError(forawrdP(inp), exp)
    print(DcDa, "dcda")
    
    DaDw = divActivation(weight, bais, inp)
    print(DaDw, "DaDw")
    
    DcDw = DcDa*DaDw
    print(DcDw)
    weight -= DcDw*learningRate
    #print()

print(forawrdP(0.8), ":forward, Error:", calculateError(forawrdP(0.8),1))
print(weight)


#train ingle nurons for if input > 0.5 output 1
for i in range(1,150):
    num = random.random()
    if num >= 0.5:
        pred = 1
    else:
        pred = 0
    print(num, pred)
    backProp(num, pred)

  
print(forawrdP(0.6), ":forward, Error:", calculateError(forawrdP(0.6  ),1))
print(forawrdP(0.2), ":forward, Error:", calculateError(forawrdP(0.2  ),0))



