from pydantic import BaseModel


class UserSchema(BaseModel):
    name: str
    designation: str
    age: int
