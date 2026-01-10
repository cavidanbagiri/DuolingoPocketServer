

from pydantic import BaseModel

class FriendRequestCreateSchema(BaseModel):
    receiver_id: int

