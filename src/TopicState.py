from pydantic import BaseModel
from typing import Annotated
import operator

class TopicState(BaseModel):
    topic: str
    draft: str = ""
    feedback : str = ""
    score : int = 0
    iteration: int = 0
    quality_threshold: int
    max_iteration : int
    min_iteration : int = 2
    final_status : str = ""
    history : Annotated[list[dict], operator.add] = []
    summary: str = ""


