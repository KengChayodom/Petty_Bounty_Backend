# schemas/common.py
from pydantic import BaseModel
from typing import Any, Optional

class StandardResponse(BaseModel):
    status: str
    message: str
    data: Optional[Any] = None