# for test1
from typing import List
from pydantic import BaseModel

class Student(BaseModel):
   id: int
   name: str
   subjects: List[str] = []
   
data = {
   'id': 1,
   'name': 'Ravikumar',
   'subjects': ["Eng", "Maths", "Sci"],
}

s1=Student(**data)

print(s1)

print(s1.model_dump())

print(s1.model_dump().get("id")) #.get is useful if the argument is missing (which will return None)
print(s1.model_dump()["id"])
