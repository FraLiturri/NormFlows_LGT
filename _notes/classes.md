
Classes are a user-defined template for creating objects. When creating a new class, we define a new type of object. 

Here's a list of the main topics.

- [Creating a class](#creating-a-class)
- [init() function](#init-function)
- [Self](#self)
- [Class var. and instance var.](#class-var-and-instance-var)
- [Getter and Setter methods](#getter-and-setter-methods)
- [Method overriding](#method-overriding)
- [Static and class methods](#static-and-class-methods)
- [Overload decorator](#overload-decorator)
- [Abstract classes and methods](#abstract-classes-and-methods)
- [getattr(object, name, default) method](#getattrobject-name-default-method)
- [The call method](#the-call-method)


## Creating a class
```python
#Class creation
class Dog: 
    sound = "bark" #class attribute; 

dog1 = Dog() #create an object dog1 of type Dog(); 
print(dog1.sound) #accessing attribute; 
```


## init() function
Automatically initialzes object attributed when it's created. The `__init__()` method can be seen as the constructor in C++. 

```python
class Dog: 
    species = "Canine" #class attribute; 

    def __init__(self, name, age): 
    """initializes name and age attributes: 
    in such a way, when defining a new object of type 
    Dog, two arguments are requested.""" 
        self.name = name #here the var. "age" passed through the __init__() method is saved under the attribute self.name; 
        self.age = age #same as before; 

dog1 = Dog("Buddy", 3) #name: Budddy, age: 3; 
print(dog1.name)
print(dog1.species)
```
## Self 
Self parameter is a reference to the current istance of the class: can be used to access attributes and methods of th eobject. 

```python
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def bark(self):
        print(f"{self.name} is barking!") #self.name to access dog's name; 

dog1 = Dog("Buddy", 3)
dog1.bark()
```
## Class var. and instance var.
Class variables are shared among all the instances of the class, whereas instance variables are related only to a specific instance of the class. 

```python
class Dog: 
    species = "Canine" #Class var.;

    def __init__(self, name, age): #instance var: name and age are defined when an object is created; 
        self.name = name
        self.age = age

dog1 = Dog("Buddy", 3)
dog2 = Dog("Charlie", 5)

print(dog1.species) #accessing to class var.; 
print(dog1.name) #instance var.; 

dog1.name = "Max" #Now the instance is updated; 
Dog.species = "Feline" #Class var. is updated; 
```

## Getter and Setter methods
Getter is used to access the value of an attribute, setter to modify it. 
`@property` is a decorator which allows a more elegant way to define a var. and to get it. 

*Note*: the underscore behind a var. means the variable has to be considerate private, so any programmer shouldn't work on it. Instead the getter and setter methods are used to access and modify the value. 

```python 
class Dog:
    def __init__(self, name, age):
        self._name = name  # Conventionally private variable
        self._age = age  # Conventionally private variable

    @property
    def name(self):
        return self._name  # Getter

    @name.setter
    def name(self, value):
        self._name = value  # Setter

    @property
    def age(self):
        return self._age  # Getter

    @age.setter
    def age(self, value):
        if value < 0:
            print("Age cannot be negative!")
        else:
            self._age = value  # Setter

```

## Method overriding
This occurs when a subclass provides a specific implementation of a method defined before in its superclass; 

``` python
class Animal:
    def sound(self):
        print("Some sound")

class Dog(Animal): #Dog is a subclass of Animal (note that there's no limit on the number of parents that a subclass can have); 
    @override #can be omitted, however to avoid silent typo it's better to use the decorator; 
    def sound(self):  # Method overriding: specifing the sound; 
        print("Woof")

dog = Dog()
dog.sound()
```
## Static and class methods
Static method hasn't access to the class and it's defined using the decorator `@staticmethod`. On the other side Class method has access to the class through the `cls` parameter. Here's an example. 

```python 
class Dog:
    @staticmethod
    def info():
        print("Dogs are loyal animals.")

    @classmethod
    def count(cls):
        print("There are many dogs of class", cls)

dog = Dog()
dog.info()  # Static method call
dog.count() # Class method call
```

## Overload decorator
`@overload` is used to manage type hints and define different behaviours of a certain function depending on the inputs type; here's an example:

```python
from typing import overload

@overload
def sum(a: int, b: int) -> int: ...
@overload 
def sum(a: NDArray, b:NDArray) -> NDArray: ...

def sum(a,b):
    if is_instance(a, int) and is_instance(b, int):
        return a+b
    else:
        return [a[i] + b[i] for i in range(len(a))] #let's assume len(a) = len(b); 

res = sum(2, 4)
```
in this case if a different argument (from `int` or `float`) is passed, the IDE will return an error. It's important to notice that 
`@overload` is used for easy-type checking on the IDE, not during the runtime. 


## Abstract classes and methods
Abstract classes provide a template for other classes (cannot be instanciated directly, since it's abstract, so the `__init__()` method hasn't to be called); they contain abstract methods, which definitions is passed to the subclasses, where the methods has to be overrided. 

```python
from abc import ABC, abstractmethod #tools for abstract classes and much more!; 

class Animal(ABC): #abstract base class
    @abstractmethod
    def sound(self):
        pass #method is passed (equivalently, also "..." can be used); 

class Dog(Animal):
    def sound(self): #overriding; 
        print("Woof")

dog = Dog()
dog.sound()
```

## getattr(object, name, default) method
This function gets the attribute of an object, if this doesn't exists returns the default value. 

```python
class Person:
    name = "Alice"

person = Person()

getattr(person, 'name', None) #returns True; 
getattr(person, 'age', None) #returns None (as specified if the attributes doesn't exist); 
```

## The call method
This is used to call a method of a class without specifing it; let's see an example: 

```python 
class Person: 
    def presentation(self, name, age: int, city):
        print(f'Hi, my name is {name}, I am {age} years old. I live in {city}.')

    def __call__(self, *args, **kwargs): 
        return self.presentation(*args, **kwargs) #the function to call; 

Bob = Person()
Bob("Bob", 24, city = "Rome") #without __call__(), should be: Bob.presentation("Bob", 24, city = "Rome"); 
```
in this case the `__call__()` method passes the parameters specified as arguments to the object Bob to `presentation`. 
This make also the code more readable, improving clarity. 

Note: `kwargs` indicates all the parameters passed with a key when the function/method is called. 

