# schemas/common.py
from pydantic import BaseModel
from typing import Any, Optional

class StandardResponse(BaseModel):
    status: str
    message: str
    data: Optional[Any] = None


class PaginatedData(BaseModel):
    """The `data` payload of a listing endpoint.

    `total` is the number of rows matching the filter, independent of `limit`
    and `offset`, so a client can render numbered pages and a queue depth. A
    listing that returned rows alone would leave both to guesswork: a full page
    is not evidence that another page exists.
    """

    items: list[Any]
    total: int
    limit: int
    offset: int