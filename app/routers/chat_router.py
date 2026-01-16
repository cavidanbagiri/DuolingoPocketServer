

from typing import List, Optional
import httpx

from fastapi import APIRouter, HTTPException, status, Query
from fastapi.params import Depends

from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.token_handler import TokenHandler
from app.database.setup import get_db

from app.repositories.chat_repository import (ChatConversationRepository,
                                              CreateConversationRepository,
                                              ChatConversationGetConversationByIdRepository,
                                              GetFriendsListRepository,
                                              SendFriendRequestRepository,
                                              FetchUsersRepository,
                                              GetUserByIdRepository,
                                              GetFriendRequestsRepository,
                                                FriendRequestActionRepository
                                              )

from app.schemas.chat_schemas import FriendRequestCreateSchema

from app.logging_config import setup_logger

router = APIRouter()

logger = setup_logger(__name__, "chat.log")


# In your FastAPI backend
@router.get("/conversations")
async def get_conversations(
        user_info: dict = Depends(TokenHandler.verify_access_token),
        db: AsyncSession = Depends(get_db)
    ):
    """
    Return user Conversations
    :param user_info:
    :param db:
    :return:
    """
    try:
        repo = ChatConversationRepository(
            db = db,
            user_id=int(user_info.get('sub')), # Get User Id
        )
        result = await repo.get_conversations()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during chat get conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )

@router.post("/conversations")
async def create_conversation(
        participant_ids: List[int],
        user_info: dict = Depends(TokenHandler.verify_access_token),
        db: AsyncSession = Depends(get_db)
):
    """
    Create a new conversation
    :param participant_ids:
    :param user_info:
    :param db:
    :return:
    """
    try:
        repo = CreateConversationRepository(
            db=db,
            user_id=int(user_info.get('sub')),
            participant_ids = participant_ids
        )

        result = await repo.create_conversation()

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during chat create conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )


@router.get("/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: int,
        user_info: dict = Depends(TokenHandler.verify_access_token),
        db: AsyncSession = Depends(get_db)
                       ):
    """
    Return all messages for a conversation
    :param conversation_id:
    :param user_info:
    :param db:
    :return:
    """
    try:
        repo = ChatConversationGetConversationByIdRepository(db=db,
                                                             user_id=int(user_info.get('sub')),
                                                             conversation_id=conversation_id)
        result = await repo.get_messages()
        print('the result is {}'.format(result))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during chat get conversation by id: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )




# FastAPI - Friend requests endpoints - Fetch all friends
@router.get("/friends")
async def get_friends(
    user_info: dict = Depends(TokenHandler.verify_access_token),
    db: AsyncSession = Depends(get_db)
):
    """Get user's friends list"""
    try:
        repo = GetFriendsListRepository(db=db,user_id=int(user_info.get('sub')))
        result = await repo.get_friends()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during fething friends list: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )



# Send request for friends
@router.post("/friends/requests")
async def send_friend_request(
    request_data: FriendRequestCreateSchema,
    user_info: dict = Depends(TokenHandler.verify_access_token),
    db: AsyncSession = Depends(get_db)
):
    """Send friend request"""
    try:
        repo = SendFriendRequestRepository(
            db=db,
            user_id=int(user_info.get('sub')),
            receiver_id=request_data.receiver_id
        )
        result = await repo.send_friend_request()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during send friend request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )



# Fetch all friend requests
@router.get("/friends/requests")
async def get_friend_requests(
    user_info: dict = Depends(TokenHandler.verify_access_token),
    db: AsyncSession = Depends(get_db)
):
    """
    Get pending friend requests
    :param user_info:
    :param db:
    :return:
    """
    try:
        repo = GetFriendRequestsRepository(
            db=db,
            user_id=int(user_info.get('sub'))
        )
        result = await repo.get_friend_requests()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during get friend requests: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )



# Accept friend request
@router.post("/friends/requests/{request_id}/accept")
async def accept_friend_request(
    request_id: int,
    user_info: dict = Depends(TokenHandler.verify_access_token),
    db: AsyncSession = Depends(get_db)
):
    """Accept a friend request"""
    try:
        repo = FriendRequestActionRepository(
            db=db,
            user_id=int(user_info.get('sub')),
            request_id=request_id
        )
        result = await repo.accept_request()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during accept friend request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )


# Reject friend request
@router.post("/friends/requests/{request_id}/reject")
async def reject_friend_request(
    request_id: int,
    user_info: dict = Depends(TokenHandler.verify_access_token),
    db: AsyncSession = Depends(get_db)
):
    """Reject a friend request"""
    try:
        repo = FriendRequestActionRepository(
            db=db,
            user_id=int(user_info.get('sub')),
            request_id=request_id
        )
        result = await repo.reject_request()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during reject friend request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )



# Fetch or search the users
@router.get('/fetch_users')
async def fetch_users(
    search: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    user_info: dict = Depends(TokenHandler.verify_access_token),
    db: AsyncSession = Depends(get_db)
):
    """Fetch users to show in frontend"""
    try:
        repo = FetchUsersRepository(
            db=db,
            user_id=int(user_info.get('sub')),
            search=search,
            limit=limit
        )
        result = await repo.fetch_users()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during fetch users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )


# Get information about the user
@router.get('/get_user_by_id')
async def get_user_by_id(
        getting_user_id: int,
        user_info: dict = Depends(TokenHandler.verify_access_token),
        db: AsyncSession = Depends(get_db)
):
    """
    :param user_id:
    :param user_info:
    :param db:
    :return:
    """

    try:
        repo = GetUserByIdRepository(db=db, user_id=int(user_info.get('sub')), getting_user_id = getting_user_id )
        result = await repo.get_user_by_id()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during accept friend request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred")



#
# @router.post("/friends/requests/{request_id}/reject")
# async def reject_friend_request(
#     request_id: int,
#     user_info: dict = Depends(TokenHandler.verify_access_token),
#     db: AsyncSession = Depends(get_db)
# ):
#     """Reject friend request"""
#     try:
#         repo = RejectFriendRequestRepository(
#             db=db,
#             user_id=int(user_info.get('sub')),
#             request_id=request_id
#         )
#         result = await repo.reject_friend_request()
#         return result
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error during reject friend request: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="An unexpected error occurred"
#         )
#
#
# @router.delete("/friends/{friend_id}")
# async def remove_friend(
#     friend_id: int,
#     user_info: dict = Depends(TokenHandler.verify_access_token),
#     db: AsyncSession = Depends(get_db)
# ):
#     """Remove friend"""
#     try:
#         repo = RemoveFriendRepository(
#             db=db,
#             user_id=int(user_info.get('sub')),
#             friend_id=friend_id
#         )
#         result = await repo.remove_friend()
#         return result
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error during remove friend: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="An unexpected error occurred"
#         )