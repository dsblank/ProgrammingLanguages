# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.1'
#       jupytext_version: 0.8.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.4
# ---

# %% [markdown]
# # 1. Scheme in Python

# %% [markdown]
# ## 1.1 Complete with Continuations, Nondeterminsitic search, and No Stack Limits

# %% [markdown]
# This notebook describes the basic techniques of implementing Scheme in Python in Continuation-Passing Style (CPS), and call-with-current-continuation.
#
# **Goal**: explore implementing the core of a real Scheme in Python in about 50 cells.
#
# First, we need to write some parsing code so that Python will be able to read Scheme's [symbolic expressions](https://en.wikipedia.org/wiki/S-expression), also called "s-expressions".
#
# We start at the lowest level, taking a string and return a list of strings broken up based on white space and brackets.

# %%
def lexer(string):
    retval = []
    current = ""
    for i in range(len(string)):
        if string[i] in ["(", "[", ")", "]"]:
            if current:
                retval.append(current)
            current = ""
            retval.append(string[i])
        elif string[i] in [" ", "\t", "\n"]:
            if current:
                retval.append(current)
            current = ""
        else:
            current += string[i]
    if current:
        retval.append(current)
    return retval

# %%
lexer("(this is a (test) of 1 [thing])")

# %% [markdown]
# The output of `lexer` is just a list of parts. We now pass that to `reader` which will create lists, and turn strings of numbers into actual numbers:

# %%
def reader(lexp):
    current = None
    stack = []
    for item in lexp:
        if item.isdigit():
            if current is not None:
                current.append(eval(item))
            else:
                current = eval(item)
        elif item in ["[", "("]:
            if current is not None:
                stack.append(current)
            current = []
        elif item in ["]", ")"]:
            if stack:
                stack[-1].append(current)
                current = stack.pop(-1)
            else:
                pass
        else:
            if current is not None:
                current.append(item)
            else:
                current = item
    return current

# %%
reader(lexer("(this is a (test) of 1 or 2 [things])"))

# %% [markdown]
# If we wanted, we could also parse quoted strings and floating-point numbers in `reader`. We'll add those later.

# %% [markdown]
# Now, we are ready to construct s-expressions. Our goal is to feed the output of reader into parser, and have the parser create [Abstract Syntax Tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree).
#
# First, we define in Python all of the functions and classes that we will need to process s-expressions, including: `car`, `cdr`, and `cons`:

# %%
class Symbol(str):
    "Class to define symbols, which should be unique items"
    def __repr__(self):
        return str.__repr__(self)[1:-1] # don't show quotation marks

EmptyList = Symbol("()")
void = Symbol("<void>")

class Cons(object):
    "A cell/link used to construct linked-list"
    def __init__(self, car, cdr):
        self.car = car
        self.cdr = cdr
    def __repr__(self):
        if self.car == "closure-exp":
            return "#<procedure %s>" % self.cdr.cdr.car
        elif self.car == "primitive-exp":
            return "#<primitive %s>" % self.cdr.car
        retval = ""
        current = self
        while current is not EmptyList and isinstance(current, Cons):
            if retval != "":
                retval += " "
            retval += str(current.car) # recursion here!
            current = current.cdr
        if current is not EmptyList:
            retval += " . " + str(current)
        return "(%s)" % retval

class String(str):
    "Class to wrap strings so that we can define repr"
    def __repr__(self):
        return '"%s"' % str.__repr__(self)[1:-1]
    def __str__(self):
        return self.__repr__()

def List(*args):
    "Create a linked-list of items"
    retval = EmptyList
    for arg in reversed(args):
        retval = cons(arg, retval)
    return retval

def car(exp):
    return exp.car
def cadr(exp):
    return exp.cdr.car
def caddr(exp):
    return exp.cdr.cdr.car
def cadddr(exp):
    return exp.cdr.cdr.cdr.car

def cdr(exp):
    return exp.cdr
def cddr(exp):
    return exp.cdr.cdr
def cdddr(exp):
    return exp.cdr.cdr.cdr
def cddddr(exp):
    return exp.cdr.cdr.cdr.cdr

def cons(item1, item2):
    return Cons(item1, item2)

def length(item, handler):
    current = item
    count = 0
    while current is not EmptyList and isinstance(current, Cons):
        current = current.cdr
        count += 1
    if current is not EmptyList:
        return handler("Attempt to take length of improper list")
    return count

def rac(item):
    current = item
    previous = None
    while current is not EmptyList:
        previous = current.car
        current = current.cdr
    return previous

## We use "_q" in Python to represent "?" in Scheme.

def pair_q(item):
    return isinstance(item, Cons)

def null_q(lst):
    return lst is EmptyList

# %% [markdown]
# Note that we create classes for Symbols and Cons cells. We use the Symbol class to both make unique items, and to be able to represent Symbols as we wish. We use the Cons class to create linked-lists made of cons cells, and to be able to format these as Scheme lists.
#
# Now, we construct a set of tagged-lists so that the parser can identify each of the "special forms".

# %%
def lit_exp(value):
    return List("lit-exp", value)

def if_exp(test_exp, then_exp, else_exp):
    return List("if-exp", test_exp, then_exp, else_exp)

def let_exp(variables, values, body):
    return List("let-exp", variables, values, body)

def var_exp(symbol):
    return List("var-exp", symbol)

def assign_exp(symbol, value):
    return List("assign-exp", symbol, value)

def define_exp(symbol, value):
    return List("define-exp", symbol, value)

def app_exp(f, args):
    return List("app-exp", f, args)

def procedure_exp(variables, body):
    return List("procedure-exp", variables, body)

def closure_exp(env, variables, body):
    return List("closure-exp", env, variables, body)

def begin_exp(exps):
    return List("begin-exp", exps)

## and a utility function for defining closure?:
def closure_q(item):
    return pair_q(item) and car(item) == "closure-exp"

def primitive_q(item):
    return pair_q(item) and car(item) == "primitive-exp"

# %% [markdown]
# And now the parser:

# %%
def sexp(item):
    """
    Takes an Python list of items and returns Scheme s-exp.
    """
    if isinstance(item, list):
        return List(*map(sexp, item)) # recursion!
    else:
        return item

def parser(pyexp):
    """
    Reads in a Python list of things, and returns a sexp.
    Uses Python stack for recursion.
    """
    if isinstance(pyexp, int):
        return lit_exp(pyexp)
    elif isinstance(pyexp, str):
        return var_exp(pyexp)
    elif pyexp[0] == "quote":
        return lit_exp(sexp(pyexp[1]))
    elif pyexp[0] == "set!":
        return assign_exp(pyexp[1], parser(pyexp[2]))
    elif pyexp[0] == "define":
        return define_exp(pyexp[1], parser(pyexp[2]))
    elif pyexp[0] == "begin":
        return begin_exp(List(*map(parser, pyexp[1:])))
    elif pyexp[0] == "if":
        if len(pyexp) == 3:
            return if_exp(parser(pyexp[1]), parser(pyexp[2]), lit_exp(EmptyList))
        else:
            return if_exp(parser(pyexp[1]), parser(pyexp[2]), parser(pyexp[3]))
    elif pyexp[0] == "let":
        return let_exp(List(*[pair[0] for pair in pyexp[1]]),
                       List(*map(parser, [pair[1] for pair in pyexp[1]])),
                       parser(pyexp[2]))
    elif pyexp[0] == "lambda":
        return procedure_exp(List(*pyexp[1]), parser(pyexp[2]))
    else:
        return app_exp(parser(pyexp[0]), List(*map(parser, pyexp[1:])))

# %% [markdown]
# The parser is so short! That is mostly because of Scheme's prefix notation.
#
# Let's try parsing a few expressions:

# %%
parser(reader(lexer("(let ((x 1)(y 2)) x)")))

# %% [markdown]
# Now, let's add the pieces necessary to build an interpreter.
#
# We need to be able to represent boolean values. For now, we will use 0 to represent #f (False). Anything else is #t (True). Note that this is different from Python's idea of true/false. For example, in Python [] and (,) are considered false.

# %%
def true_q(value):
    if value == 0:
        return False
    else:
        return True

# %% [markdown]
# An environment will be a Python list of [variable, value] pairs. Later we can add [Lexial Address](https://en.wikipedia.org/wiki/Scope_(computer_science%29) to make lookup [O(1)](https://en.wikipedia.org/wiki/Big_O_notation).
#
# When we extend an environment (when we evaluate a `let` or `lambda`) we will make a new "frame"... a list of variable/value pairs append onto the front of an existing environment:

# %%
def list_to_vector(lst):
    retval = []
    current = lst
    while current is not EmptyList:
        retval.append(current.car)
        current = current.cdr
    return retval

def extend_env(vars, vals, env):
    # list is needed at both levels to allow value changes:
    return [list(map(list,zip(list_to_vector(vars),
                              list_to_vector(vals)))), env]

# %%
list(map(list, zip([1, 2, 3], [4, 5, 6])))

# %% [markdown]
# When we look up a variable, we will actually return the [variable, value] pair (or return False):

# %%
def lookup_binding(symbol, env, first_frame_only=False):
    frame = env[0]
    for pair in frame:
        if pair[0] == symbol:
            return pair
    if not first_frame_only and len(env) > 1:
        return lookup_binding(symbol, env[1])
    else:
        return False

# %% [markdown]
# And now we are ready to define the interpreter.
#
# We write the interpreter, here called `evaluator`, in [Continuation Passing Style](https://en.wikipedia.org/wiki/Continuation-passing_style), or CPS. This version is still recursive, but this won't have any impact on our Scheme recursive programs. Our Scheme functions won't use Python's stack---eventually.
#
# First, we wrote the evaluator in regular form. Then we converted it to CPS. Note that:
#
# 1. We use a third and fourth parameter to evaluator (handler and k) that represent the **error and computation continuations**, respectively
# 1. We use **Python functions** (closures) to **represent continuations**
# 1. We pass **results** to the continuation via **apply**
# 1. Any place where there is still computation to be done, we construct a **new continuation** using Python's **lambda**
#
# To see more examples of converting Python code into Continuation Passing Style, see:
#
# * [Review, Continuations and CPS](http://jupyter.cs.brynmawr.edu/hub/dblank/public/Review,%20Continuations%20and%20CPS.ipynb)

# %%
def apply(f, *args):
    return f(*args)

import operator
from goto import with_goto

@with_goto
def main(exp_reg, env_reg, handler_reg, k_reg, start):
    label .start
    if start == "evaluator":
        goto .evaluator
    elif start == "add_prim":
        goto .add_prim
    elif start == "pair_q_prim":
        goto .pair_q_prim
    elif start == "error_prim":
        goto .error_prim
    elif start == "null_q_prim":
        goto .null_q_prim
    elif start == "map_prim":
        goto .map_prim
    elif start == "car_prim":
        goto .car_prim
    elif start == "cdr_prim":
        goto .cdr_prim
    elif start == "add_prim":
        goto .add_prim
    elif start == "sub_prim":
        goto .sub_prim
    elif start == "mul_prim":
        goto .mul_prim
    elif start == "div_prim":
        goto .div_prim
    elif start == "equal_sign_prim":
        goto .equal_sign_prim
    elif start == "less_than_prim":
        goto .less_than_prim
    elif start == "greater_than_prim":
        goto .greater_than_prim
    elif start == "cons_prim":
        goto .cons_prim
    elif start == "print_prim":
        goto .print_prim
    elif start == "input_prim":
        goto .input_prim
    elif start == "length_prim":
        goto .length_prim
    elif start == "list_prim":
        goto .list_prim
    elif start == "callcc_prim":
        goto .callcc_prim
    else:
        raise Exception("invalid start: %s" % start)

    ##################################
    # exp_reg, env_reg, handler_reg, k_reg
    # def evaluator(exp, env, handler, k):
    label .evaluator
    if car(exp_reg) == "lit-exp":
        k_reg, value_reg = k_reg, cadr(exp_reg)
        goto .apply_cont
    elif car(exp_reg) == "var-exp":
        binding = lookup_binding(cadr(exp_reg), env_reg)
        if not binding:
            raise Exception("variable not defined: %s" % cadr(exp_reg))
        k_reg, value_reg = k_reg, binding[1]
        goto .apply_cont
    elif car(exp_reg) == "let-exp":
        variables = cadr(exp_reg)
        exp_reg, env_reg, handler_reg, k_reg = (caddr(exp_reg), env_reg, handler_reg,
                     ["cont1", exp_reg, variables, env_reg, handler_reg, k_reg])
        goto .evaluator_map
    elif car(exp_reg) == "if-exp":
        exp_reg, env_reg, handler_reg, k_reg = (cadr(exp_reg), env_reg, handler_reg,
                     ["cont2", exp_reg, env_reg, handler_reg, k_reg])
        goto .evaluator
    elif car(exp_reg) == "assign-exp":
        variable = cadr(exp_reg)
        exp_reg, env_reg, handler_reg, k_reg = (caddr(exp_reg), env_reg, handler_reg,
                     ["cont3", variable, env_reg, handler_reg, k_reg])
        goto .evaluator
    elif car(exp_reg) == "define-exp":
        variable = cadr(exp_reg)
        exp_reg, env_reg, handler_reg, k_reg = (caddr(exp_reg), env_reg, handler_reg,
                     ["cont4", variable, env_reg, handler_reg, k_reg])
        goto .evaluator
    elif car(exp_reg) == "procedure-exp":
        k_reg, value_reg = k_reg, closure_exp(env_reg, cadr(exp_reg), caddr(exp_reg))
        goto .apply_cont
    elif car(exp_reg) == "app-exp":
        exp_reg, env_reg, handler_reg, k_reg = (cadr(exp_reg), env_reg, handler_reg,
                     ["cont6", exp_reg, env_reg, handler_reg, k_reg])
        goto .evaluator
    elif car(exp_reg) == "begin-exp":
        exp_reg, env_reg, handler_reg, k_reg = (cadr(exp_reg), env_reg, handler_reg,
                     ["cont7", k_reg])
        goto .evaluator_map
    else:
        handler("invalid abstract syntax: %s" % exp)
        goto .exit


    ##################################
    # value_reg, variable_reg, env_reg, handler_reg, k_reg
    #def eval_assign(value, variable, env, handler, k):
    label .eval_assign
    binding = lookup_binding(variable_reg, env_reg)
    if binding:
        binding[1] = value_reg
    else:
        handler("No such variable: '%s'" % variable_reg)
        goto .exit
    k_reg, value_reg = k_reg, value_reg
    goto .apply_cont

    ##################################
    # value_reg, variable_reg, env_reg, handler_reg, k_reg
    # def eval_define(value, variable, env, handler, k):
    label .eval_define
    binding = lookup_binding(variable_reg, env_reg, True)
    if binding:
        binding[1] = value_reg
    else:
        env_reg[0].append([variable_reg, value_reg])
    k_reg, value_reg = k_reg, None
    goto .apply_cont

    ##################################
    # value_reg, exp_reg, env_reg, handler_reg, k_reg
    # def eval_if(value, exp, env, handler, k):
    label .eval_if
    if true_q(value_reg):
        exp_reg, env_reg, handler_reg, k_reg = (caddr(exp_reg), env_reg, handler_reg, k_reg)
        goto .evaluator
    else:
        exp_reg, env_reg, handler_reg, k_reg = (cadddr(exp_reg), env_reg, handler_reg, k_reg)
        goto .evaluator

    ##################################
    # exp_reg, env_reg, handler_reg, k_reg
    #def evaluator_map(exp, env, handler, k):
    label .evaluator_map
    if null_q(exp_reg):
        k_reg, value_reg = k_reg, EmptyList
        goto .apply_cont
    else:
        exp_reg, env_reg, handler_reg, k_reg = (car(exp_reg), env_reg, handler_reg,
                     ["cont9", exp_reg, env_reg, handler_reg, k_reg])
        goto .evaluator

    ##################################
    # f_reg, args_reg, env_reg, handler_reg, k_reg
    # def eval_app(f, args, env, handler, k):
    label .eval_app
    if closure_q(f_reg):
        env = cadr(f_reg)
        parameters = caddr(f_reg)
        body = cadddr(f_reg)
        exp_reg, env_reg, handler_reg, k_reg = (body, extend_env(parameters, args_reg, env_reg), handler_reg, k_reg)
        goto .evaluator
    elif primitive_q(f_reg):
        start = cadr(f_reg) ## primitive name
        goto .start
    else:
        raise Exception("invalid application")

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    # def add_prim(args, env, handler, k):
    label .add_prim
    k_reg, value_reg = k_reg, operator.add(*list_to_vector(args_reg))
    goto .apply_cont

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    #def sub_prim(args, env, handler, k):
    label .sub_prim
    k_reg, value_reg = k_reg, operator.sub(*list_to_vector(args_reg))
    goto .apply_cont

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    #def mul_prim(args, env, handler, k):
    label .mul_prim
    k_reg, value_reg = k_reg, operator.mul(*list_to_vector(args_reg))
    goto .apply_cont

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    #def div_prim(args, env, handler, k):
    label .div_prim
    k_reg, value_reg = k_reg, operator.div(*list_to_vector(args_reg))
    goto .apply_cont

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    #def equal_sign_prim(args, env, handler, k):
    label .equal_sign_prim
    a = car(args_reg)
    b = cadr(args_reg)
    k_reg, value_reg = k_reg, 1 if a == b else 0
    goto .apply_cont

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    #def less_than_prim(args, env, handler, k):
    label .less_than_prim
    a = car(args_reg)
    b = cadr(args_reg)
    k_reg, value_reg = k_reg, 1 if a < b else 0
    goto .apply_cont

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    #def greater_than_prim(args, env, handler, k):
    label .greater_than_prim
    a = car(args_reg)
    b = cadr(args_reg)
    k_reg, value_reg = k_reg, 1 if a > b else 0
    goto .apply_cont

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    #def cons_prim(args, env, handler, k):
    label .cons_prim
    k_reg, value_reg = k_reg, cons(car(args_reg), cadr(args_reg))
    goto .apply_cont

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    #def print_prim(args, env, handler, k):
    label .print_prim
    print(*list_to_vector(args_reg))
    k_reg, value_reg = k_reg, void
    goto .apply_cont

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    #def input_prim(args, env, handler, k):
    label .input_prim
    prompt = ""
    if length(args_reg, handler_reg) > 0:
        prompt = car(args_reg)
    retval = raw_input(prompt)
    k_reg, value_reg = k_reg, retval
    goto .apply_cont

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    #def length_prim(args, env, handler, k):
    label .length_prim
    k_reg, value_reg = k_reg, length(car(args_reg), handler_reg)
    goto .apply_cont

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    #def list_prim(args, env, handler, k):
    label .list_prim
    k_reg, value_reg = k_reg, args_reg
    goto .apply_cont

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    #def not_prim(args, env, handler, k):
    label .not_prim
    k_reg, value_reg = k_reg, 0 if true_q(car(args_reg)) else 1
    goto .apply_cont

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    #def pair_q_prim(args, env, handler, k):
    label .pair_q_prim
    k_reg, value_reg = k_reg, 1 if pair_q(car(args_reg)) else 0
    goto .apply_cont

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    #def error_prim(args, env, handler, k):
    label .error_prim
    apply(handler_reg, car(args_reg))
    goto .exit

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    #def null_q_prim(args, env, handler, k):
    label .null_q_prim
    k_reg, value_reg = k_reg, 1 if null_q(car(args_reg)) else 0
    goto .apply_cont

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    #def car_prim(args, env, handler, k):
    label .car_prim
    k_reg, value_reg = k_reg, car(car(args_reg))
    goto .apply_cont

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    #def cdr_prim(args, env, handler, k):
    label .cdr_prim
    k_reg, value_reg = k_reg, cdr(car(args_reg))
    goto .apply_cont

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    #def map_prim(args, env, handler, k):
    label .map_prim
    f = car(args_reg)
    args = cadr(args_reg)
    if null_q(args_reg):
        k_reg, value_reg = k_reg, args_reg
        goto .apply_cont
    else:
        f_reg, args_reg, env_reg, handler_reg, k_reg = (f, args, env_reg, handler_reg,
                        ["cont11", f, args, env_reg, handler_reg, k_reg])
        goto .eval_app

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    #def for_each_prim(args, env, handler, k):
    label .for_each_prim
    f = car(args_reg)
    args = cadr(args_reg)
    if null_q(args):
        k_reg, value_reg = k_reg, args
        goto .apply_cont
    else:
        f_reg, args_reg, env_reg, handler_reg, k_reg = (f, args, env_reg, handler_reg,
                        ["cont13", f, args, env_reg, handler_reg, k_reg])
        goto .eval_app

    ##################################
    # args_reg, env_reg, handler_reg, k_reg
    #def callcc_prim(args, env, handler, k):
    label .callcc_prim
    proc = car(args_reg)
    ## FIXME: what should this be?
    fake_k = lambda args, env, handler, k2: apply_cont(k_reg, car(args_reg))
    f_reg, args_reg, env_reg, handler_reg, k_reg = (proc, List(fake_k), env_reg, handler_reg, k_reg)
    goto .eval_app

    ##################################
    # k_reg, value_reg
    #def apply_cont(k, value):
    label .apply_cont
    if k_reg[0] == "cont0":
        goto .retval
    elif k_reg[0] == "cont1":
        exp, variables, env, handler, k = k_reg[1:]
        exp_reg, env_reg, handler_reg, k_reg = cadddr(exp), extend_env(variables, value_reg, env), handler, k
        goto .evaluator
    elif k_reg[0] == "cont2":
        exp, env, handler, k = k_reg[1:]
        value_reg, exp_reg, env_reg, handler_reg, k_reg = value_reg, exp, env, handler, k
        goto .eval_if
    elif k_reg[0] == "cont3":
        variable, env, handler, k = k_reg[1:]
        value_reg, variable_reg, env_reg, handler_reg, k_reg = value_reg, variable, env, handler, k
        goto .eval_assign
    elif k_reg[0] == "cont4":
        variable, env, handler, k = k_reg[1:]
        value_reg, variable_reg, env_reg, handler_reg, k_reg = value_reg, variable, env, handler, k
        goto .eval_define
    elif k_reg[0] == "cont5":
        v, env, handler, k = k_reg[1:]
        f_reg, args_reg, env_reg, handler_reg, k_reg = v, value_reg, env, handler, k
        goto .eval_app
    elif k_reg[0] == "cont6":
        exp, env, handler, k = k_reg[1:]
        exp_reg, env_reg, handler_reg, k_reg = (caddr(exp), env, handler,
                                                ["cont5", value_reg, env, handler, k])
        goto .evaluator_map
    elif k_reg[0] == "cont7":
        k = k_reg[1]
        k_reg, value_reg = k, rac(value_reg)
        goto .apply_cont
    elif k_reg[0] == "cont8":
        v1, k = k_reg[1:]
        k_reg, value_reg = k, cons(v1, value_reg)
        goto .apply_cont
    elif k_reg[0] == "cont9":
        exp, env, handler, k = k_reg[1:]
        exp_reg, env_reg, handler_reg, k_reg = (cdr(exp), env, handler,
                                                ["cont8", value_reg, k])
        goto .evaluator_map
    elif k_reg[0] == "cont10":
        v, k = k_reg[1:]
        k_reg, value_reg = k, cons(v, value_reg)
        goto .apply_cont
    elif k_reg[0] == "cont11":
        f, args, env, handler, k = k_reg[1:]
        args_reg, env_reg, handler_reg, k_reg = (List(f, cdr(args)), env, handler,
                                                 ["cont10", value_reg, k])
        goto .map_prim
    elif k_reg[0] == "cont12":
        v, k = k_reg[1:]
        k_reg, value_reg = k_reg, cons(v, value_reg)
        goto .apply_cont
    elif k_reg[0] == "cont13":
        f, args, env, handler, k = k_reg[1:]
        args_reg, env_reg, handler_reg, k_reg = (List(f, cdr(args)), env, handler,
                                                 ["cont12", value_reg, k])
        goto .map_prim
    else:
        raise Exception("unknown continuation: %s" % (k_reg,))

    ##################################
    label .exit
    value_reg = None
    label .retval
    return value_reg

toplevel_env = [[
    ["()", EmptyList],
    ["not", List("primitive-exp", "not_prim")],
    ["pair?", List("primitive-exp", "pair_q_prim")],
    ["error", List("primitive-exp", "error_prim")],
    ["null?", List("primitive-exp", "null_q_prim")],
    ["map", List("primitive-exp", "map_prim")],
    ["car", List("primitive-exp", "car_prim")],
    ["cdr", List("primitive-exp", "cdr_prim")],
    ["+", List("primitive-exp", "add_prim")],
    ["-", List("primitive-exp", "sub_prim")],
    ["*", List("primitive-exp", "mul_prim")],
    ["/", List("primitive-exp", "div_prim")],
    ["=", List("primitive-exp", "equal_sign_prim")],
    ["<", List("primitive-exp", "less_than_prim")],
    [">", List("primitive-exp", "greater_than_prim")],
    ["cons", List("primitive-exp", "cons_prim")],
    ["print", List("primitive-exp", "print_prim")],
    ["input", List("primitive-exp", "input_prim")],
    ["length", List("primitive-exp", "length_prim")],
    ["list", List("primitive-exp", "list_prim")],
    ["call/cc", List("primitive-exp", "callcc_prim")],
]]

# %% [markdown]
# Note that all of the primitives are defined using CPS as well. That allows us to implement exotic functions, such as `call-with-current-continuation` (call/cc) as shown.

# %% [markdown]
# Finally, we define a short-cut for running programs:

# %%
def scheme(exp):
    return main(
        exp_reg=parser(reader(lexer(exp))),
        env_reg=toplevel_env,
        handler_reg = lambda error: error,
        k_reg = ["cont0"],
        start="evaluator"
    )

# %% [markdown]
# And now, we are ready to compute!

# %%
scheme("(cons 1 2)")

# %%
scheme("(+ 1 2)")

# %%
scheme("(length (list 1 2))")

# %%
scheme("(if (+ 1 2) 6 7)")

# %%
scheme("((lambda (n) (+ n 1)) 42)")

# %%
scheme("(if 0 2 3)")

# %%
scheme("(if 1 2 3)")

# %%
scheme("(+ 1 (error 23))")

# %%
scheme("(print 12 56)")

# %%
scheme("""
(begin
   (print 1 3)
   (+ 2 3)
   (+ 3 3)
   4)
""")

# %%
scheme("(map (lambda (n) (print n)) (quote (1 2 3 4)))")

# %%
scheme("(> 1 2)")

# %%
scheme("""

(begin
  (define x 1)
  (print x)
)

""")

# %%
scheme("""

(begin
  (define f
     (lambda (n)
        (* n n)))

  (print (f 5))
)

""")

# %% [markdown]
# ## 1.2 Python's Stack

# %% [markdown]
# We can define global recursive functions:

# %%
scheme("(define factorial (lambda (n) (if (= n 1) 1 (* n (factorial (- n 1))))))")

# %%
scheme("factorial")

# %%
scheme("(factorial 1)")

# %%
scheme("(factorial 5)")

# %% [markdown]
# Unfortunately, we are still using Python's stack. So this fails:

# %%
# try:
#     scheme("(factorial 1000)")
# except:
#     print("Crashed Python's stack!")

# %% [markdown]
# ## 1.5 Getting rid of Python's stack

# %% [markdown]
# It turns out to be very easy to stop using Python's stack for our function calls. We will delay executing continuations. So, instead of apply k to v, instead we just package it up.
#
# Recall how we have defined these functions:
#
# ```python
# def apply(f, *args):
#     return f(*args)
#
# def scheme(exp):
#     return evaluator(parser(reader(lexer(exp))),
#                      toplevel_env,
#                      lambda error: error,
#                      lambda v: v)
# ```
#
# We then implement a very simple trampoline that will execute the continuation. This won't crash Python's stask because we execute each continuation outside of the recursive calls:
