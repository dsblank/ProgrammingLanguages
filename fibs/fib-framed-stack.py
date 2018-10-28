# fib register machine with framed continuation stack

import collections

class Stack(collections.deque):
    def push(self, x):
        self.appendleft(x)
    def pop(self):
        return self.popleft()

n_reg = None
k_reg = None
value_reg = None
pc_reg = None

def make_frame0():
    return ["frame0"]

def make_frame1(n):
    return ["frame1", n]

def make_frame2(v1):
    return ["frame2", v1]

def fib(n):
    global n_reg, k_reg, value_reg, pc_reg
    k_reg = Stack()
    n_reg = n
    k_reg.push(make_frame0())
    pc_reg = fib_cps
    loop()
    return value_reg

def loop():
    while pc_reg:
        pc_reg()

def fib_cps():
    global n_reg, k_reg, value_reg, pc_reg
    if n_reg < 3:
        value_reg = 1
        pc_reg = apply_cont
    else:
        k_reg.push(make_frame1(n_reg))
        n_reg = n_reg - 1
        pc_reg = fib_cps

def apply_cont():
    global n_reg, k_reg, value_reg, pc_reg
    frame = k_reg.pop()
    if frame[0] == "frame0":
        pc_reg = None
    elif frame[0] == "frame1":
        n_reg = frame[1] - 2
        k_reg.push(make_frame2(value_reg))
        pc_reg = fib_cps
    elif frame[0] == "frame2":
        value_reg = frame[1] + value_reg
        pc_reg = apply_cont
    else:
        raise Exception("error in car")
