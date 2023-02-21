from typing import List
from pydantic import BaseModel


class GenerateItem(BaseModel):
    title: str
    color: str
    desc: str

    def get_prompt(self):
        return f"title: {self.title}\ncolor: {self.color}\nstyle: {self.desc}"


class EditItem(BaseModel):
    image: str
    source: GenerateItem
    target: GenerateItem


class ResponseItem(BaseModel):
    images: List[str]
