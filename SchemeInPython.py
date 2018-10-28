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
#     version: 3.6.5
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

# %%
def evaluator(exp, env, handler, k):
    if car(exp) == "lit-exp":
        return apply(k, cadr(exp))
    elif car(exp) == "var-exp":
        binding = lookup_binding(cadr(exp), env)
        if not binding:
            return apply(handler, "variable not defined: %s" % cadr(exp))
        return apply(k, binding[1])
    elif car(exp) == "let-exp":
        variables = cadr(exp)
        return apply(evaluator_map, caddr(exp), env, handler,
                 lambda values: evaluator(cadddr(exp), 
                     extend_env(variables, values, env), handler, k))
    elif car(exp) == "if-exp":
        return apply(evaluator, cadr(exp), env, handler,
                 lambda value: eval_if(value, exp, env, handler, k))
    elif car(exp) == "assign-exp":
        variable = cadr(exp)
        return apply(evaluator, caddr(exp), env, handler,
                 lambda value: eval_assign(value, variable, env, handler, k))
    elif car(exp) == "define-exp":
        variable = cadr(exp)
        return apply(evaluator, caddr(exp), env, handler,
                 lambda value: eval_define(value, variable, env, handler, k))
    elif car(exp) == "procedure-exp":
        return apply(k, closure_exp(env, cadr(exp), caddr(exp)))
    elif car(exp) == "app-exp":
        return apply(evaluator, cadr(exp), env, handler,
             lambda v: evaluator_map(caddr(exp), env, handler,
                lambda all: eval_app(v, all, env, handler, k)))
    elif car(exp) == "begin-exp":
        return apply(evaluator_map, cadr(exp), env, handler,
                  lambda all: apply(k, rac(all)))
    else:
        return handler("invalid abstract syntax: %s" % exp)

        
def eval_assign(value, variable, env, handler, k):
    binding = lookup_binding(variable, env)
    if binding:
        binding[1] = value
    else:
        return handler("No such variable: '%s'" % variable)
    return apply(k, value)

def eval_define(value, variable, env, handler, k):
    binding = lookup_binding(variable, env, True)
    if binding:
        binding[1] = value
    else:
        env[0].append([variable, value])
    return apply(k, None)        
        
def eval_if(value, exp, env, handler, k):
    if true_q(value):
        return apply(evaluator, caddr(exp), env, handler, k)
    else:
        return apply(evaluator, cadddr(exp), env, handler, k)

def evaluator_map(exp, env, handler, k):
    if null_q(exp):
        return apply(k, EmptyList)
    else:
        return apply(evaluator, car(exp), env, handler,
                  lambda v1: evaluator_map(cdr(exp), env, handler,
                      lambda rest: apply(k, cons(v1, rest))))
    
def eval_app(f, args, env, handler, k):
    if closure_q(f):
        env = cadr(f)
        parameters = caddr(f)
        body = cadddr(f)
        return apply(evaluator, body, extend_env(parameters, args, env), handler, k)
    else: 
        return f(args, env, handler, k)

# %% [markdown]
# Now, we construct a toplevel environment with some primitives, like +, -, \*, print, etc. Primitives are written in CPS.

# %%
import operator

def add_prim(args, env, handler, k):
    return apply(k, operator.add(*list_to_vector(args)))

def sub_prim(args, env, handler, k):
    return apply(k, operator.sub(*list_to_vector(args)))

def mul_prim(args, env, handler, k):
    return apply(k, operator.mul(*list_to_vector(args)))

def div_prim(args, env, handler, k):
    return apply(k, operator.div(*list_to_vector(args)))

def equal_sign_prim(args, env, handler, k):
    a = car(args)
    b = cadr(args)
    return apply(k, 1 if a == b else 0)

def less_than_prim(args, env, handler, k):
    a = car(args)
    b = cadr(args)
    return apply(k, 1 if a < b else 0)

def greater_than_prim(args, env, handler, k):
    a = car(args)
    b = cadr(args)
    return apply(k, 1 if a > b else 0)

def cons_prim(args, env, handler, k):
    return apply(k, cons(car(args), cadr(args)))

def print_prim(args, env, handler, k):
    print(*list_to_vector(args))
    return apply(k, void)
    
def input_prim(args, env, handler, k):
    prompt = ""
    if length(args, handler) > 0:
        prompt = car(args)
    retval = raw_input(prompt)
    return apply(k, retval)
    
def length_prim(args, env, handler, k):
    return apply(k, length(car(args), handler))

def list_prim(args, env, handler, k):
    return apply(k, args)

def not_prim(args, env, handler, k):
    return apply(k, 0 if true_q(car(args)) else 1)

def pair_q_prim(args, env, handler, k):
    return apply(k, 1 if pair_q(car(args)) else 0)

def error_prim(args, env, handler, k):
    return apply(handler, car(args))

def null_q_prim(args, env, handler, k):
    return apply(k, 1 if null_q(car(args)) else 0)

def car_prim(args, env, handler, k):
    return apply(k, car(car(args)))

def cdr_prim(args, env, handler, k):
    return apply(k, cdr(car(args)))

def map_prim(args, env, handler, k):
    f = car(args)
    args = cadr(args)
    if null_q(args):
        return apply(k, args)
    else:
        return eval_app(f, args, env, handler,
                lambda v: map_prim(List(f, cdr(args)), env, handler, 
                    lambda rest: apply(k, cons(v, rest))))

def for_each_prim(args, env, handler, k):
    f = car(args)
    args = cadr(args)
    if null_q(args):
        return apply(k, args)
    else:
        return eval_app(f, args, env, handler,
                lambda v: map_prim(List(f, cdr(args)), env, handler, 
                    lambda rest: apply(k, cons(v, rest))))

def callcc_prim(args, env, handler, k):
    proc = car(args)
    fake_k = lambda args, env, handler, k2: apply(k, car(args))
    return eval_app(proc, List(fake_k), env, handler, k)

toplevel_env = [[["()", EmptyList],
                 ["not", not_prim],
                 ["pair?", pair_q_prim],
                 ["error", error_prim],
                 ["null?", null_q_prim],
                 ["map", map_prim],
                 ["car", car_prim],
                 ["cdr", cdr_prim],
                 ["+", add_prim], 
                 ["-", sub_prim],
                 ["*", mul_prim],
                 ["/", div_prim],
                 ["=", equal_sign_prim],
                 ["<", less_than_prim],
                 [">", greater_than_prim],
                 ["cons", cons_prim],
                 ["print", print_prim],
                 ["input", input_prim],
                 ["length", length_prim],
                 ["list", list_prim],
                 ["call/cc", callcc_prim]]] 

# %% [markdown]
# Note that all of the primitives are defined using CPS as well. That allows us to implement exotic functions, such as `call-with-current-continuation` (call/cc) as shown.

# %% [markdown]
# Finally, we define a short-cut for running programs:

# %%
def scheme(exp):
    return evaluator(parser(reader(lexer(exp))), 
                     toplevel_env, 
                     lambda error: error, 
                     lambda v: v)

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
try:
    scheme("(factorial 1000)")
except:
    print("Crashed Python's stack!")

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

# %%
## Use Python's list as continuation wrapper:

def apply(k, *v):
    return ["<continuation>", k, v]

def continuation_q(exp):
    return (isinstance(exp, list) and 
            len(exp) > 0 and
            exp[0] == "<continuation>")

def trampoline(result):
    while continuation_q(result):
        f = result[1]
        args = result[2]
        result = f(*args)
    return result

def scheme(exp):
    return trampoline(evaluator(parser(reader(lexer(exp))), toplevel_env, lambda e: e, lambda v: v))

# %% [markdown]
# Now, we can evaluate much deeper recursive calls than Python:

# %%
scheme("(factorial 1000)")

# %%
evaluator(parser(reader(lexer("factorial"))), toplevel_env, lambda e: e, lambda v: v)

# %%
trampoline(evaluator(parser(reader(lexer("factorial"))), toplevel_env, lambda e: e, lambda v: v))

# %% [markdown]
# ## Running Forever

# %% [markdown]
# In fact, this would run forever:
#
# ```python
# scheme("""
# (define loop
#     (lambda ()
#         (loop)))
# (loop)
# """)
# ```

# %% [markdown]
# This loop doesn't use up memory as it runs... once started, it has zero additional impact. Why?

# %% [markdown]
# ## Efficiency

# %% [markdown]
# How do the runtimes compare between Python and Scheme-in-Python?

# # %%
# %%timeit
# scheme("(factorial 900)")

# %%
def fact(n):
    if n == 1:
        return 1
    else:
        return n * fact(n - 1)

# # %%
# %%timeit
# fact(900)

# %% [markdown]
# So, the Scheme-in-Python runs about **2 magnitudes slower** than Python for a program that has lots of recursive function calls. That's not good, but there are some things we can do to improve that a bit. But for many programmers, this is a price worth paying for the additional power (e.g., continuations, call/cc, and no stack limit).

# %% [markdown]
# ## 1.3 Call/cc
#
#
# `call-with-current-continuation` is a function that takes a function (f) and gives it the computation left to do.
#
# Consider attempting to implement return as in the following example:

# %%
scheme("""
(define f 
  (lambda (return)
    (begin
      (return 2)
      3)))""")

# %%
scheme("""
(print (f (lambda (x) x))) 
""")

# %% [markdown]
# Nothing a regular programming language would allow you to implement getting 2 back from this function.
#
# But, with call/cc:

# %%
scheme("""
(print (call/cc f)) 
""")

# %% [markdown]
# ### 1.3.1 Continuations as First Class Objects

# %% [markdown]
# Our Scheme language uses continuations, but we can't really take advantage of them unless we can actually grab them. call-with-current-continuation allows us to grab them.
#
# We define a function called `current-continuation` that uses call/cc to simply return the current continuation:

# %% [markdown]
# Consider the following:
#
# ```
# (call/cc (lambda (cc) (cc cc)))
# ```

# %%
scheme("""
(begin 
    (print 1)
    (print 2)
    (print 3)
    (define resume (call/cc (lambda (cc) cc)))
    (print 4)
    (print 5)
    (print 6)
)
""")

# %%
scheme("""
(resume resume)
""")

# %%
scheme("""
(define current-continuation
   (lambda ()
    (call/cc (lambda (cc)
                cc))))
""")

# %%
scheme("""
(begin 
    (print 1)
    (print 2)
    (print 3)
    (define resume (current-continuation))
    (print 4)
    (print 5)
    (print 6)
)
""")

# %% [markdown]
# But, we can go back in time and run the code starting at the point where we grabbed the continuation:

# %%
scheme("(resume resume)")

# %% [markdown]
# And do it again and again and ...

# %%
scheme("(resume resume)")

# %% [markdown]
# ## 1.4 Amb - nondeterministic programming

# %% [markdown]
# Now that we have call/cc we can implement the [amb operator](https://en.wikipedia.org/wiki/Nondeterministic_programming) (called "choose" in Calysto Scheme).
#
# We will keep track of decision points in a `fail-stack`. Each time we "fail", we will go back to that point in the computation and resume computing again, with the next choice.

# %%
scheme("(define fail-stack (quote ()))")

scheme("""
(define fail
   (lambda ()
    (if (not (pair? fail-stack))
        (error (quote ()))
        (begin
          (let ((back-track-point (car fail-stack)))
            (begin
             (set! fail-stack (cdr fail-stack))
             (back-track-point back-track-point)))))))
""")

scheme("""
(define amb 
   (lambda (choices)
    (let ((cc (current-continuation)))
      (if (null? choices)      
          (fail)
          (if (pair? choices)      
              (let ((choice (car choices)))
                 (begin 
                   (set! choices (cdr choices))
                   (set! fail-stack (cons cc fail-stack))
                   choice)))))))
""")

scheme("""
(define assert 
   (lambda (condition)
    (if (not condition)
        (fail)
        1)))
""")

# %%
scheme("(amb (list 1 2 3))")

# %%
scheme("(fail)")

# %%
scheme("(fail)")

# %%
scheme("(fail)")

# %% [markdown]
# ## Pythagorean Mystery

# %% [markdown]
# We're looking for dimensions of a legal right triangle using the Pythagorean theorem:
#
# $ a^2 + b^2 = c^2 $
#
# And, we want the second side (b) to be the shorter one:
#
# $ b < a $
#
# We can express the problem as three variables and two asserts:

# %%
scheme("(define fail-stack (quote ()))")

# %%
scheme("""
(let ((a (amb (list 1 2 3 4 5 6 7)))
      (b (amb (list 1 2 3 4 5 6 7)))
      (c (amb (list 1 2 3 4 5 6 7))))
  (begin
    (assert (= (* c c) (+ (* a a) (* b b))))
    (assert (< b a))
    (print (list a b c))
  )
)
""")

# %% [markdown]
# ## 1.6 Summary

# %% [markdown]
# We have examined how one could implement the "hard parts" of Scheme in Python. But there are still more things to do:
#
# * Error checking: to make the concepts easy to see, we left out most error checking. We should add that in a manner that doesn't slow down processing too much.
# * Add [Lexical Address](https://mitpress.mit.edu/sicp/full-text/sicp/book/node131.html): use indexing to get variable values rather than sequential search.
# * Feedback when there is an error: we could easily add stack-like traces to this Scheme that look just like Python's. That would address one of [Guido's concerns](http://neopythonic.blogspot.com/2009/04/tail-recursion-elimination.html). Read that, compare to this code, and you'll see why Guido is wrong.
# * There is still recursion (and thus stack use) in the interpreter: we need to get rid of the evaluator and sexp recursions. We can do that by turning recursive calls into a type of continuation, or rewriting parts in iterative forms. Of course, our Scheme will remain stack-free. 
# * Macros: we will need to write a `define-syntax`. But that is an independent function performed in the parser.
# * Integrate with Python. This is easy: call Python functions, call Scheme functions from Python, import Python modules.
#
# There are many subtleties left to discover! You may want to start with this code and add to it to get to a complete Scheme implementation. Try the below exercises to get started.
#
# All of the above (and more!) have been done in [Calysto Scheme](https://github.com/Calysto/calysto/blob/master/calysto/language/scheme.py). However, Calysto Scheme was written in Scheme and converted to Python (similar to the above) via an automated process (well, 1k lines of Python written by hand, and 8k lines of Python generated by [Calico Scheme](https://bitbucket.org/ipre/calico/src/master/languages/Scheme/Scheme/)). You can read more about the project [here](http://calicoproject.org/Calico_Scheme).
#
# We hope that this was useful!

# %% [markdown]
# # 2. Exercises

# %% [markdown]
# Try implementing the following (roughly listed from easiest to hardest):
#
# 1. Test all of the special forms. Report any errors to <mailto:dblank@cs.brynmawr.edu>.
# 1. Add error checking. For example, currently if you attempt to take the car of a non-pair or access undefined variables, you get an ugly error. Use the `handler` to return errors.
# 1. Add double-quoted strings to the language.
# 1. Add floating-point numbers to the language.
# 1. Add missing functions, like: atom?.
# 1. Add `eq?` that uses " obj1 is obj2" to compare objects.
# 1. Add a Boolean type/class and use it rather than zero/one for True/False. Make it show as #t/#f.
# 1. Add quote symbol to the parser so that 'item expands to (quote item).
# 1. Add `trace-lambda` to the language to help with debugging.
# 1. Add a stack-trace to show "stack-like" output when there is an error (eg, keep track of app-exp entrance and exit).
# 1. Add `cond` special form to the language. You will need `else` (which could be defined to be True).
# 1. Try some of your homework solutions to see if the version of Scheme in Python can do everything that Calysto Scheme can do. If not, add it!

# %% [markdown]
# <div id="disqus_thread"></div>
#     <script type="text/javascript">
#         var disqus_shortname = 'calicoproject'; // required: replace example with your forum shortname
#         (function() {
#             var dsq = document.createElement('script'); dsq.type = 'text/javascript'; dsq.async = true;
#             dsq.src = '//' + disqus_shortname + '.disqus.com/embed.js';
#             (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(dsq);
#         })();
#     </script>
#     <noscript>Please enable JavaScript to view the <a href="http://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>
#     <a href="http://disqus.com" class="dsq-brlink">comments powered by <span class="logo-disqus">Disqus</span></a>
