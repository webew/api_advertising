from pydantic import BaseModel

class Inputdata(BaseModel):
    tv: float
    radio: float
    newspaper: float