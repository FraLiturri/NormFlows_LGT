
# Abstract Base Classes (ABC)

Abstract classes are useful to check if a certain object has a certain attribute, method or property.
Of course `hasattr()` and `isinstance` can be used, however checking a lot of properties in such a way can be inconvenient. 
The main goal of the abstract classes is to provide a standardized way to test whether an object respects a given specification, to prevent the instanciation of a subclass that doesn't override a particular method in the superclass. 

- [Abstract Base Classes (ABC)](#abstract-base-classes-abc)
- [Declaring an ABC](#declaring-an-abc)
- [Why declare an ABC?](#why-declare-an-abc)
- [Subclasshook method](#subclasshook-method)
- [The abc.abstractmethod](#the-abcabstractmethod)
- [Abstract Properties](#abstract-properties)
- [Sequence and MutableSequence class](#sequence-and-mutablesequence-class)
- [Iterator class](#iterator-class)
- [Type management](#type-management)
  - [Type statement](#type-statement)
  - [TypeVar()](#typevar)
  - [Generic](#generic)
- [Self](#self)
- [Recall](#recall)
  - [hasattr(obj, key) function](#hasattrobj-key-function)
  - [isinstance(obj, classinfo)](#isinstanceobj-classinfo)

# Declaring an ABC
Fist of all it's necessary to understand the role of ABCMeta metaclass: this provides the method `register`, callable by its instance; using the register method, any abstract class can become an ancestor of any concrete class. 

```python
import abc

class AbstractClass(metaclass = abc.ABCMeta)
    def abstract_function(self):
        return None

print(AbstractClass.register(dict)) #in such a way now dict is a subclass of AbstractClass; 
print(issubclass(dict, AbstractClass)) #True; 
```

# Why declare an ABC? 
Problem: what if you want to accept as input a list of objects, without any restriction on their type? 
Checking with `isinstance` for tuples and lists (as elements of the main list) it's not enough, since there isn't any restriction, as said, on the elements types.
This problem can be solved by creating an object and registering it with the `register` method. 

```python
import abc

class MySequence(metaclass=abc.ABCMeta):
    pass

class CustomListLikeObjCls(object):
    pass

MySequence.register(CustomListLikeObjCls) #now CustomLink is a subclass of MySequence; 
print(issubclass(CustomListLikeObjCls, MySequence))
```
the same can be done using `register` as a decorator: 

```python
import abc


class MySequence(metaclass=abc.ABCMeta):
    pass

@MySequence.register #decorator: the following function is registered to the specified one;
class CustomListLikeObjCls(object):
    pass

print(issubclass(CustomListLikeObjCls, MySequence))
```
# Subclasshook method
What about subclassing (creating a subclass from another) basing on a particular method? In the following we are gonna define an absstract class and make sure that all its subclasses have the hookmethod, specified in `getattr()`.

All this has to be defined as a class method using `@classmethod` decorator, which executes the function underneath everytime the class is called. It takes one additional argument other than the class (cls) and can return True, False or NotImplemented. 

```python 
import abc

class AbstractClass(metaclass=abc.ABCMeta):
    @classmethod #thanks to this decorator the following function runs everytime the AbstractClass is called; 
    def __subclasshook__(cls, other):
        print('subclass hook:', other)
        hookmethod = getattr(other, 'hookmethod', None)
        return callable(hookmethod) #in this way we select only the callable object, not variables or non-callable obejcts;

class SubClass(object):
    def hookmethod(self):
        pass

class NormalClass(object):
    hookmethod = 'hook'

print(issubclass(SubClass, AbstractClass)) #True; 
print(issubclass(NormalClass, AbstractClass)) #False: NormalClass doesn't contain hookmethod; 
```
# The abc.abstractmethod
What about avoid instanciating a subclass that doesn't override a certain method in the superclass? This can be performed with the `abc.abstractmethod`: 

```python
import abc

class AbstractClass(metaclass=abc.ABCMeta):
    @abc.abstractmethod #decorator for override forcing; 
    def abstractName(self):
        pass

class InvalidSubClass(AbstractClass):
    pass

isc = InvalidSubClass() #since doesn't override abstractName: this returns an error!; 
```
# Abstract Properties
We can use `@property` and `abc.abstractmethod` decorators to declare properties of an abstract class: 

```python
import abc

class AbstractClass(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def abstractName(self):
        pass

class ValidSubClass(AbstractClass):
    @property
    def abstractName(self):
        return 'Abstract 1'


vc = ValidSubClass()
print(vc.abstractName)
```
the `@property` decorator takes (as all the decorators) a callable and returns an attribute! As can be seen, the method `abstracName` has been accessed through `class.attribute` (if it were a callable, then `class.method()` would be used). Thanks to this property, we can enforce the overriding of a method when instantiating a subclass. 


# Sequence and MutableSequence class
`Sequence` and `MutableSequence` are classes representing a sequence of ordered objects; the first is used for read-only sequences, whereas the second for updatable sequences. The elements in a Sequence (whatever type it is) can be accessed through an index.

An example of Sequence are `str`, `list`, `range`. 

```python
from collections.abc import Sequence

def mean(numbers: Sequence[int]) -> int:
    return sum(numbers)/len(Sequence)

print(mean([1,2,3,4,10,22,65]))
```
The advantage of this feature is its flexibility; let's consider the same example as before, but with a list as an input:

```python 
from collections.abc import Sequence

def mean(numbers: list[int]) -> int:
    return sum(numbers)/len(Sequence)

print(mean([1,2,3,4,10,22,65])) #ok
print(mean((1,2,3,4,10,22,65))) #*NOT* ok! tuple should be converted to list. This problem wouldn't erase using Sequence (tuples are a Sequence); 
```

# Iterator class
An `Iterator`, formally, is an abstract class producing elements from a collection. 
Key properties: 
- monouse: once reached the last element, the iterator is unusable,
- stateful: keeps the current position,
- lazyness: returns the elements only on-demand. 

the last point is very important: if we define an Iterator type instead of a list, it'll use less memory. 

```python
squares_list = [x**2 for x in range(10**6)] #all the squares are produced now; 
squares_iter = (x**2 for x in range(10**6)) #doesn't calculate anything, it'll return the number only when requested; 
```

The Iterator class can be imported through `collections.abc`: 
```python
from collections.abc import Iterator
```

Notice that `list`, for example, is an **iterable**, not an iterator, hoever an iterable can be convertend into a iterator through the `iter()` command.  

# Type management

## Type statement
The `type` statement is used to define a new type alias
```python 
type Vector = list[float] #defining the new type Vector as a list of floats; 

def vec_sum(a: Vector, b: Vector) -> Vector:
    assert len(a) == len(b)
    return [a[i] + b[i] for i in range(len(a))]
```

## TypeVar()
It's possible to define a new var's type with `TypeVar()`, here's an example:
```python
from typing import TypeVar

T = TypeVar("T", int, float, str) #possible types for T; 
def generic_fun(var: T) -> T:
    return NotImplemented
```
to restric the type of T accepted from a function use

```python
def not_too_generic[T: int, float](var: T) -> T:
    return NotImplemented
```

## Generic
Used to defain a class or a container with some content: 
```python
from typing import TypeVar, Generic

T = TypeVar("T")
Class Storage(Generic[T]):
    def __init__(self, content: T):
        self.content = content
    
    def get_content(self) -> T: 
        return self.content
```
# Self
Self is used to return an instance of the class specified:
```python
from typing import Self

class LatticeModel:
    def __init__(self, beta: float):
        self.beta = beta

    def set_beta(self, val: float) -> Self:
        self.beta = val
        return self 

class U1Model(LatticeModel):
    def get_topo_charge(self) -> int:
        return 0

model = U1Model(2.0).set_beta(2.0)
charge = model.get_topo_charge() #thnaks to Self, set_beta can be called; 
``` 

---
# Recall
Before diving deeper in ABC, it's useful to know `hasattr()` and `isinstance()` functions.

## hasattr(obj, key) function
This function is uesd if an object has a certain attribute, and returns True or False. 
It has two parameters: 
- obj: The object's attribute to be checked; 
- key: The attribute that needs to be checked. 

```python
class Birds:
    wings = True

phoenix = Birds()
print("Does birds have wings?" + str(hasattr(phoenix, 'wings'))) #checks if birds has an attribute called name, returns True;
print("Does birds have gills?" + str(hasattr(phoenix, 'gills'))) #checks if birds has an attribute called gills, returns False;
```

## isinstance(obj, classinfo)
`isinstance(obj, classinfo)` is a built-in function that checks whether an object or var. is an instance of a specified type or class. It's useful to assure safe operations in dynamic coding. 
The parameters are: 
- obj: the object to check,
- classinfo: the class type,
  
returns True or False, obviously. 
Here's is an implementation: 

```python
a = 5 
b = [i, for i in range(0, 10)]

print(isinstance(a, int)) #return True: a is an int; 
print(isinstance(b, list)) #True
print(isinstance(a, bool)) #False
print(isinstance(b, (int, list, str))) #True: b is a list, which is one of the types in the tuple (int, list, str).

#The same for classes: 
class Animal: 
    pass

bird = Animal()

print(isinstance(bird, Animal) #True
```