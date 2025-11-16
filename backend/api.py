from pydantic import BaseModel

class Body(BaseModel):
    message: str