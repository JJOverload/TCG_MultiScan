def division(x:int, y:int) -> int:
   return (x//y)
a=division(10,2)
print (a)
#b=division(5,2.5)
b=division(5,2)
print (b)
c=division(5,10)
print (c)


x: int = 3
y: float = 3.14
nm: str = 'abc'
married: bool = False
names: list = ['a', 'b', 'c']
marks: tuple = (10, 20, 30)
marklist: dict = {'a': 10, 'b': 20, 'c': 30}


from typing import List, Tuple, Dict
# following line declares a List object of strings.
# If violated, mypy shows error
cities: List[str] = ['Mumbai', 'Delhi', 'Chennai']
# This is Tuple with three elements respectively
# of str, int and float type)
employee: Tuple[str, int, float] = ('Ravi', 25, 35000)
# Similarly in the following Dict, the object key should be str
# and value should be of int type, failing which
# static type checker throws error
marklist2: Dict[str, int] = {'Ravi': 61, 'Anil': 72}