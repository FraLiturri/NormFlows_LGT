# A bit of this, a bit of that

Here are collected useful code pills, functions or anything that doesn't fall under a specific category. 

- [A bit of this, a bit of that](#a-bit-of-this-a-bit-of-that)
  - [Type hints](#type-hints)
  - [zip and unzip](#zip-and-unzip)
  - [\*args and \*\*kwargs](#args-and-kwargs)
  - [warnings](#warnings)

## Type hints

A type hint, as can be deduced, is a specification of a parameter's type, requested or expected by a function or its result.

```python
def sum_calc(x : list[float] | float) -> float:
    res = 0
    for i in range(0, len(x)): 
        res += x[i]
    
    return res
```

Here `sum_calc` expects a list (of floats) or a single float and returns a float. In case of improper argument or return type, an error will be raised. 

## zip and unzip

The `zip(*iterables)` command is used to combine two or more iterables into a single iterator of tuples. Each tuple contains elements corresponding to the same index in the input iterables. 

```python

a = [1,2,3]
b = ["a", "b", "c"]

a_zip = zip(a)
ab_zip = zip(a,b)

print(list(a_zip)) #prints [(1, ), (2, ), (3, )]
print(list(ab_zip)) #prints [(1, "a"), (2, "b"), (3, "c")]
```
Python stops pairing as soon as the shortest iterable runs out of elements. 

```python
a = [('Apple', 10), ('Banana', 20), ('Orange', 30)]
fruits, quantities = zip(*a) 
```
in such a way, now, fruits and quantities are unzipped. 


## *args and **kwargs

`*args` allows to pass any number of positional arguments to a function; this arguments will be collected in a tuple, which means we can loop through them.

Example: 
```python
employers = []
def office(*args):
    for arg in args:
       employers.append(arg)


office("Mario", "Antonio", "Roberto") 
print(employers)
```

`**kwargs` allows to pass any number of keyword arguments that will be collected into a dictionary.

Example:

```python 
def fun(**kwargs):
    for k, val in kwargs.items():
        print(k, "=", val)

fun(s1='Python', s2='is', s3='Awesome')
```

## warnings
`warnings` library provides some useful tools to warn the devoloper of situations that aren't necessary exceptions. A warning, tipically, is not critical and it occurs to when there's some obsolete class, function or object. 

```python
import warnings 

warnings.warn("Warning!")
``` 