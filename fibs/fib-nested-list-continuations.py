# fib register machine with continuations as nested lists

n_reg = None
k_reg = None
value_reg = None
pc_reg = None

def make_cont0():
    return ["cont0"]

def make_cont1(n, k):
    return ["cont1", n, k]

def make_cont2(v1, k):
    return ["cont2", v1, k]

def fib(n):
    global n_reg, k_reg, value_reg, pc_reg
    n_reg = n
    k_reg = make_cont0()
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
        k_reg = make_cont1(n_reg, k_reg)
        n_reg = n_reg - 1
        pc_reg = fib_cps

def apply_cont():
    global n_reg, k_reg, value_reg, pc_reg
    if k_reg[0] == "cont0":
        pc_reg = None
    elif k_reg[0] == "cont1":
        n_reg = k_reg[1] - 2
        k_reg = make_cont2(value_reg, k_reg[2])
        pc_reg = fib_cps
    elif k_reg[0] == "cont2":
        value_reg = k_reg[1] + value_reg
        k_reg = k_reg[2]
        pc_reg = apply_cont
    else:
        raise Exception("error in car")

    
