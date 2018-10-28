# fib register machine with unframed continuation stack

import collections

class DequeStack(collections.deque):
    def push(self, x):
        self.appendleft(x)
    def pop(self):
        return self.popleft()

class VectorStack():
    def __init__(self):
        self.contents = [0]*100
        self.next_pos = 0
    def push(self, x):
        self.contents[self.next_pos] = x
        self.next_pos += 1
    def pop(self):
        if self.next_pos == 0:
            raise Exception("cannot pop an empty stack")
        self.next_pos -= 1
        return self.contents[self.next_pos]

n_reg = None
k_reg = None
value_reg = None
pc_reg = None

def fib(n):
    global n_reg, k_reg, value_reg, pc_reg
    k_reg = DequeStack()
    #k_reg = VectorStack()
    n_reg = n
    k_reg.push("frame0")
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
        k_reg.push(n_reg)
        k_reg.push("frame1")
        n_reg = n_reg - 1
        pc_reg = fib_cps

def apply_cont():
    global n_reg, k_reg, value_reg, pc_reg
    frame_tag = k_reg.pop()
    if frame_tag == "frame0":
        pc_reg = None
    elif frame_tag == "frame1":
        n = k_reg.pop()
        n_reg = n - 2
        k_reg.push(value_reg)
        k_reg.push("frame2")
        pc_reg = fib_cps
    elif frame_tag == "frame2":
        v1 = k_reg.pop()
        value_reg = v1 + value_reg
        pc_reg = apply_cont
    else:
        raise Exception("error in car")
