



# app/repositories/chat_repository.py
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.orm import selectinload, joinedload
from fastapi import HTTPException, status

from app.models.user_model import (
    UserModel,
    Conversation,
    ConversationParticipant,
    Message,
    MessageStatus,
    UserProfile
)


from app.logging_config import setup_logger
logger = setup_logger(__name__, "chat.log")


class ChatConversationRepository:
    def __init__(self, db: AsyncSession, user_id: int, participant_ids: Optional[List[int]] = None):
        self.db = db
        self.user_id = user_id
        self.participant_ids = participant_ids or []

    async def get_conversations(self) -> List[dict]:
        """
        Get all conversations for the current user
        """
        try:
            # # Query conversations where user is a participant
            # stmt = (
            #     select(Conversation)
            #     .join(
            #         ConversationParticipant,
            #         ConversationParticipant.conversation_id == Conversation.id
            #     )
            #     .where(ConversationParticipant.user_id == self.user_id)
            #     .options(
            #         selectinload(Conversation.participants).selectinload(ConversationParticipant.user).selectinload(
            #             UserModel.profile),
            #         selectinload(Conversation.messages).order_by(Message.created_at.desc()).limit(1)
            #     )
            #     .order_by(desc(Conversation.updated_at))
            # )
            #
            # result = await self.db.execute(stmt)
            # conversations = result.scalars().all()


            # New Code
            # Query conversations where user is a participant
            stmt = (
                select(Conversation)
                .join(
                    ConversationParticipant,
                    ConversationParticipant.conversation_id == Conversation.id
                )
                .where(ConversationParticipant.user_id == self.user_id)
                .options(
                    selectinload(Conversation.participants)
                    .selectinload(ConversationParticipant.user)
                    .selectinload(UserModel.profile)
                )
                .order_by(desc(Conversation.updated_at))
            )

            result = await self.db.execute(stmt)
            conversations = result.scalars().all()




            # Format response
            formatted_conversations = []
            for conv in conversations:
                # Get other participants (excluding current user)
                other_participants = [
                    p for p in conv.participants
                    if p.user_id != self.user_id
                ]

                # Old Code Get last message
                # last_message = conv.messages[0] if conv.messages else None

                # Get last message separately
                stmt_last_msg = (
                    select(Message)
                    .where(Message.conversation_id == conv.id)
                    .order_by(desc(Message.created_at))
                    .limit(1)
                )
                result_last_msg = await self.db.execute(stmt_last_msg)
                last_message = result_last_msg.scalar_one_or_none()

                # Calculate unread count
                unread_count = 0
                if last_message:
                    stmt_unread = select(func.count(Message.id)).select_from(Message).join(
                        MessageStatus, MessageStatus.message_id == Message.id
                    ).where(
                        and_(
                            Message.conversation_id == conv.id,
                            MessageStatus.user_id == self.user_id,
                            MessageStatus.status != 'read',
                            Message.sender_id != self.user_id
                        )
                    )
                    result_unread = await self.db.execute(stmt_unread)
                    unread_count = result_unread.scalar() or 0

                conversation_data = {
                    "id": conv.id,
                    "is_group": conv.is_group,
                    "group_name": conv.group_name,
                    "group_image_url": conv.group_image_url,
                    "created_at": conv.created_at.isoformat() if conv.created_at else None,
                    "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
                    "participants": [
                        {
                            "id": p.user.id,
                            "username": p.user.username,
                            "email": p.user.email,
                            "profile": {
                                "first_name": p.user.profile.first_name if p.user.profile else None,
                                "last_name": p.user.profile.last_name if p.user.profile else None,
                                "profile_image_url": p.user.profile.profile_image_url if p.user.profile else None,
                            } if p.user.profile else None
                        }
                        for p in other_participants
                    ],
                    "last_message": {
                        "id": last_message.id,
                        "content": last_message.content,
                        "message_type": last_message.message_type,
                        "sender_id": last_message.sender_id,
                        "created_at": last_message.created_at.isoformat() if last_message.created_at else None,
                    } if last_message else None,
                    "unread_count": unread_count
                }
                formatted_conversations.append(conversation_data)

            return formatted_conversations

        except Exception as e:
            logger.error(e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error fetching conversations: {str(e)}"
            )


class CreateConversationRepository:
    def __init__(self, db: AsyncSession, user_id: int, participant_ids: List[int]):
        self.db = db
        self.user_id = user_id
        self.participant_ids = participant_ids

    async def create_conversation(self) -> dict:
        """
        Create a new conversation (1:1 or group)
        """
        try:
            # Add current user to participants
            all_participants = [self.user_id] + self.participant_ids

            # Remove duplicates
            all_participants = list(set(all_participants))

            # Check if conversation already exists (for 1:1 chats)
            if len(all_participants) == 2:
                stmt_existing = select(Conversation).join(
                    ConversationParticipant,
                    Conversation.id == ConversationParticipant.conversation_id
                ).where(
                    ConversationParticipant.user_id.in_(all_participants)
                ).group_by(Conversation.id).having(
                    func.count(ConversationParticipant.user_id) == 2
                )

                result_existing = await self.db.execute(stmt_existing)
                existing_conversation = result_existing.scalar_one_or_none()

                if existing_conversation:
                    # Return existing conversation
                    return await self._format_conversation(existing_conversation)

            # Create new conversation
            is_group = len(all_participants) > 2

            conversation = Conversation(
                is_group=is_group,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            self.db.add(conversation)
            await self.db.flush()  # Get the ID

            # Add participants
            for participant_id in all_participants:
                participant = ConversationParticipant(
                    conversation_id=conversation.id,
                    user_id=participant_id,
                    joined_at=datetime.utcnow(),
                    is_admin=(participant_id == self.user_id)  # Creator is admin
                )
                self.db.add(participant)

            await self.db.commit()

            # Fetch the complete conversation with participants
            stmt = select(Conversation).where(Conversation.id == conversation.id).options(
                selectinload(Conversation.participants).selectinload(ConversationParticipant.user).selectinload(
                    UserModel.profile)
            )
            result = await self.db.execute(stmt)
            full_conversation = result.scalar_one()

            return await self._format_conversation(full_conversation)

        except Exception as e:
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating conversation: {str(e)}"
            )


    async def _format_conversation(self, conversation: Conversation) -> dict:
        """Helper to format conversation response"""
        other_participants = [
            p for p in conversation.participants
            if p.user_id != self.user_id
        ]

        return {
            "id": conversation.id,
            "is_group": conversation.is_group,
            "group_name": conversation.group_name,
            "group_image_url": conversation.group_image_url,
            "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
            "updated_at": conversation.updated_at.isoformat() if conversation.updated_at else None,
            "participants": [
                {
                    "id": p.user.id,
                    "username": p.user.username,
                    "email": p.user.email,
                    "profile": {
                        "first_name": p.user.profile.first_name if p.user.profile else None,
                        "last_name": p.user.profile.last_name if p.user.profile else None,
                        "profile_image_url": p.user.profile.profile_image_url if p.user.profile else None,
                    } if p.user.profile else None
                }
                for p in other_participants
            ]
        }


class ChatConversationGetConversationByIdRepository:
    def __init__(self, db: AsyncSession, user_id: int, conversation_id: int):
        self.db = db
        self.user_id = user_id
        self.conversation_id = conversation_id

    async def get_messages(self) -> List[dict]:
        """
        Get all messages for a conversation
        """
        try:
            # First verify user is a participant
            stmt_participant = select(ConversationParticipant).where(
                and_(
                    ConversationParticipant.conversation_id == self.conversation_id,
                    ConversationParticipant.user_id == self.user_id
                )
            )
            result_participant = await self.db.execute(stmt_participant)
            participant = result_participant.scalar_one_or_none()

            if not participant:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not a participant in this conversation"
                )

            # Fetch messages with sender info and status
            stmt = (
                select(Message)
                .where(Message.conversation_id == self.conversation_id)
                .options(
                    joinedload(Message.sender).joinedload(UserModel.profile),
                    joinedload(Message.statuses),
                    joinedload(Message.reply_to).joinedload(Message.sender)
                )
                .order_by(Message.created_at)
                .limit(100)  # Paginate later
            )

            # result = await self.db.execute(stmt)
            # messages = result.scalars().all()
            result = await self.db.execute(stmt)
            messages = result.unique().scalars().all()

            # Format response
            formatted_messages = []
            for msg in messages:
                # Get status for current user
                user_status = next(
                    (s for s in msg.statuses if s.user_id == self.user_id),
                    None
                )

                message_data = {
                    "id": msg.id,
                    "conversation_id": msg.conversation_id,
                    "sender_id": msg.sender_id,
                    "sender": {
                        "id": msg.sender.id,
                        "username": msg.sender.username,
                        "profile": {
                            "first_name": msg.sender.profile.first_name if msg.sender.profile else None,
                            "last_name": msg.sender.profile.last_name if msg.sender.profile else None,
                            "profile_image_url": msg.sender.profile.profile_image_url if msg.sender.profile else None,
                        } if msg.sender.profile else None
                    },
                    "content": msg.content,
                    "message_type": msg.message_type,
                    "media_url": msg.media_url,
                    "media_thumbnail_url": msg.media_thumbnail_url,
                    "file_size": msg.file_size,
                    "is_edited": msg.is_edited,
                    "is_deleted": msg.is_deleted,
                    "reply_to": {
                        "id": msg.reply_to.id,
                        "content": msg.reply_to.content,
                        "sender_id": msg.reply_to.sender_id,
                        "sender_username": msg.reply_to.sender.username if msg.reply_to.sender else None,
                    } if msg.reply_to else None,
                    "status": user_status.status if user_status else "sent",
                    "read_at": user_status.read_at.isoformat() if user_status and user_status.read_at else None,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None,
                    "updated_at": msg.updated_at.isoformat() if msg.updated_at else None,
                }
                formatted_messages.append(message_data)

            return formatted_messages

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error fetching messages: {str(e)}"
            )

    async def update_last_read(self, message_id: int) -> None:
        """
        Update user's last read message in conversation
        """
        try:
            # Update participant's last_read_message_id
            stmt = select(ConversationParticipant).where(
                and_(
                    ConversationParticipant.conversation_id == self.conversation_id,
                    ConversationParticipant.user_id == self.user_id
                )
            )
            result = await self.db.execute(stmt)
            participant = result.scalar_one_or_none()

            if participant:
                participant.last_read_message_id = message_id
                await self.db.commit()

        except Exception as e:
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error updating last read: {str(e)}"
            )


# app/repositories/friend_repository.py
from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, update
from sqlalchemy.orm import selectinload
from fastapi import HTTPException, status
import datetime

from app.models.user_model import (
    UserModel,
    FriendshipRequest,
    UserProfile,
    friendship  # This is your association table
)


class GetFriendsListRepository:
    def __init__(self, db: AsyncSession, user_id: int):
        self.db = db
        self.user_id = user_id

    async def get_friends(self) -> List[Dict]:
        """
        Get user's friends list with their online status and last seen
        """
        try:
            # Query friends through friendships table
            stmt = (
                select(UserModel)
                .join(
                    friendship,
                    or_(
                        friendship.c.user_id == self.user_id,
                        friendship.c.friend_id == self.user_id
                    )
                )
                .where(
                    and_(
                        or_(
                            friendship.c.user_id == UserModel.id,
                            friendship.c.friend_id == UserModel.id
                        ),
                        UserModel.id != self.user_id,
                        friendship.c.status == 'accepted'
                    )
                )
                .options(
                    selectinload(UserModel.profile)
                )
                .order_by(UserModel.username)
            )

            result = await self.db.execute(stmt)
            friends = result.scalars().all()

            # Format response with friend info
            formatted_friends = []
            for friend in friends:
                # Determine friendship direction to get friendship info
                friendship_stmt = select(friendship).where(
                    or_(
                        and_(
                            friendship.c.user_id == self.user_id,
                            friendship.c.friend_id == friend.id
                        ),
                        and_(
                            friendship.c.user_id == friend.id,
                            friendship.c.friend_id == self.user_id
                        )
                    ),
                    friendship.c.status == 'accepted'
                )

                friendship_result = await self.db.execute(friendship_stmt)
                friendship_data = friendship_result.first()

                formatted_friends.append({
                    "id": friend.id,
                    "username": friend.username,
                    "email": friend.email,
                    "profile": {
                        "first_name": friend.profile.first_name if friend.profile else None,
                        "last_name": friend.profile.last_name if friend.profile else None,
                        "profile_image_url": friend.profile.profile_image_url if friend.profile else None,
                        "country": friend.profile.country if friend.profile else None,
                        "city": friend.profile.city if friend.profile else None,
                    } if friend.profile else None,
                    "native_language": friend.native,
                    "is_premium": friend.is_premium,
                    "friendship_created_at": friendship_data.created_at.isoformat() if friendship_data else None,
                    "is_online": False,  # Will be updated by Express WebSocket
                    "last_seen": None  # Will be updated by Express WebSocket
                })

            return formatted_friends

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error fetching friends list: {str(e)}"
            )


class SendFriendRequestRepository:
    def __init__(self, db: AsyncSession, user_id: int, receiver_id: int):
        self.db = db
        self.user_id = user_id
        self.receiver_id = receiver_id

    async def send_friend_request(self) -> Dict:
        """
        Send friend request
        Returns: Created friend request info
        """
        try:
            # Check if trying to add self
            if self.user_id == self.receiver_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot send friend request to yourself"
                )

            # Check if receiver exists
            receiver_stmt = select(UserModel).where(UserModel.id == self.receiver_id)
            receiver_result = await self.db.execute(receiver_stmt)
            receiver = receiver_result.scalar_one_or_none()

            if not receiver:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )

            # Check if already friends
            existing_friendship_stmt = select(friendship).where(
                or_(
                    and_(
                        friendship.c.user_id == self.user_id,
                        friendship.c.friend_id == self.receiver_id
                    ),
                    and_(
                        friendship.c.user_id == self.receiver_id,
                        friendship.c.friend_id == self.user_id
                    )
                )
            )

            existing_friendship_result = await self.db.execute(existing_friendship_stmt)
            existing_friendship = existing_friendship_result.first()

            if existing_friendship:
                if existing_friendship.status == 'accepted':
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Already friends"
                    )
                elif existing_friendship.status == 'pending':
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Friend request already sent"
                    )
                elif existing_friendship.status == 'blocked':
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Cannot send request to blocked user"
                    )

            # Check if there's already a pending request
            pending_request_stmt = select(FriendshipRequest).where(
                or_(
                    and_(
                        FriendshipRequest.sender_id == self.user_id,
                        FriendshipRequest.receiver_id == self.receiver_id,
                        FriendshipRequest.status == 'pending'
                    ),
                    and_(
                        FriendshipRequest.sender_id == self.receiver_id,
                        FriendshipRequest.receiver_id == self.user_id,
                        FriendshipRequest.status == 'pending'
                    )
                )
            )

            pending_request_result = await self.db.execute(pending_request_stmt)
            pending_request = pending_request_result.scalar_one_or_none()

            if pending_request:
                if pending_request.sender_id == self.user_id:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Friend request already sent"
                    )
                else:
                    # If other user already sent request, auto-accept it
                    return await self._auto_accept_existing_request(pending_request)

            # Create new friend request
            friend_request = FriendshipRequest(
                sender_id=self.user_id,
                receiver_id=self.receiver_id,
                status='pending',
                created_at=datetime.datetime.utcnow()
            )

            self.db.add(friend_request)
            await self.db.flush()

            # Get sender info
            sender_stmt = select(UserModel).where(
                UserModel.id == self.user_id
            ).options(selectinload(UserModel.profile))

            sender_result = await self.db.execute(sender_stmt)
            sender = sender_result.scalar_one()

            await self.db.commit()

            return {
                "id": friend_request.id,
                "sender": {
                    "id": sender.id,
                    "username": sender.username,
                    "profile": {
                        "first_name": sender.profile.first_name if sender.profile else None,
                        "profile_image_url": sender.profile.profile_image_url if sender.profile else None,
                    } if sender.profile else None
                },
                "receiver_id": self.receiver_id,
                "status": "pending",
                "created_at": friend_request.created_at.isoformat(),
                "message": "Friend request sent successfully"
            }

        except HTTPException:
            await self.db.rollback()
            raise
        except Exception as e:
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error sending friend request: {str(e)}"
            )

    async def _auto_accept_existing_request(self, existing_request: FriendshipRequest) -> Dict:
        """Auto-accept if the other user already sent a request"""
        # Update request status
        existing_request.status = 'accepted'
        existing_request.updated_at = datetime.datetime.utcnow()

        # Create friendship in both directions
        await self.db.execute(
            friendship.insert().values(
                user_id=self.user_id,
                friend_id=self.receiver_id,
                status='accepted',
                created_at=datetime.datetime.utcnow()
            )
        )

        await self.db.execute(
            friendship.insert().values(
                user_id=self.receiver_id,
                friend_id=self.user_id,
                status='accepted',
                created_at=datetime.datetime.utcnow()
            )
        )

        await self.db.commit()

        # Get both users info
        users_stmt = select(UserModel).where(
            UserModel.id.in_([self.user_id, self.receiver_id])
        ).options(selectinload(UserModel.profile))

        users_result = await self.db.execute(users_stmt)
        users = {user.id: user for user in users_result.scalars().all()}

        return {
            "id": existing_request.id,
            "sender": {
                "id": existing_request.sender_id,
                "username": users[existing_request.sender_id].username if existing_request.sender_id in users else None,
            },
            "receiver": {
                "id": existing_request.receiver_id,
                "username": users[
                    existing_request.receiver_id].username if existing_request.receiver_id in users else None,
            },
            "status": "accepted",
            "created_at": existing_request.created_at.isoformat(),
            "updated_at": existing_request.updated_at.isoformat(),
            "message": "Friend request auto-accepted (mutual request)"
        }


class FriendRequestActionRepository:
    def __init__(self, db: AsyncSession, user_id: int, request_id: int):
        self.db = db
        self.user_id = user_id
        self.request_id = request_id

    async def accept_request(self) -> Dict:
        """Accept a friend request"""
        try:
            # Get the request
            stmt = select(FriendshipRequest).where(
                and_(
                    FriendshipRequest.id == self.request_id,
                    FriendshipRequest.receiver_id == self.user_id,
                    FriendshipRequest.status == 'pending'
                )
            )
            result = await self.db.execute(stmt)
            request = result.scalar_one_or_none()

            if not request:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Friend request not found or already processed"
                )

            # Update request status
            request.status = 'accepted'
            request.updated_at = func.now()

            # Add friendship to both directions
            # User -> Friend

            print('////////////////////////////............................', request)

            await self.db.execute(
                friendship.insert().values(
                    user_id=request.receiver_id,
                    friend_id=request.sender_id,
                    # status='accepted',
                    created_at=func.now()
                )
            )

            # Friend -> User
            await self.db.execute(
                friendship.insert().values(
                    user_id=request.sender_id,
                    friend_id=request.receiver_id,
                    # status='accepted',
                    created_at=func.now()
                )
            )

            await self.db.commit()

            return {
                "message": "Friend request accepted",
                "request_id": request.id,
                "friend_id": request.sender_id,
                "status": request.status
            }

        except HTTPException:
            await self.db.rollback()
            raise
        except Exception as e:
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error accepting friend request: {str(e)}"
            )

    async def reject_request(self) -> Dict:
        """Reject a friend request"""
        try:
            # Get the request
            stmt = select(FriendshipRequest).where(
                and_(
                    FriendshipRequest.id == self.request_id,
                    FriendshipRequest.receiver_id == self.user_id,
                    FriendshipRequest.status == 'pending'
                )
            )
            result = await self.db.execute(stmt)
            request = result.scalar_one_or_none()

            if not request:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Friend request not found or already processed"
                )

            # Update request status
            request.status = 'rejected'
            request.updated_at = func.now()

            await self.db.commit()

            return {
                "message": "Friend request rejected",
                "request_id": request.id,
                "sender_id": request.sender_id,
                "status": request.status
            }

        except HTTPException:
            await self.db.rollback()
            raise
        except Exception as e:
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error rejecting friend request: {str(e)}"
            )



class FetchUsersRepository:

    def __init__(self, db: AsyncSession, user_id: int, search: Optional[str] = None, limit: int = 50):
        self.db = db
        self.user_id = user_id
        self.search = search
        self.limit = limit

    async def fetch_users(self) -> Dict:
        """
        Fetch users to show in frontend (excluding current user and friends)
        Returns users with friendship status
        """
        try:
            # Get current user's friends and pending requests
            friends_stmt = select(friendship.c.friend_id).where(
                friendship.c.user_id == self.user_id,
                friendship.c.status == 'accepted'
            )

            sent_requests_stmt = select(FriendshipRequest.receiver_id).where(
                FriendshipRequest.sender_id == self.user_id,
                FriendshipRequest.status == 'pending'
            )

            received_requests_stmt = select(FriendshipRequest.sender_id).where(
                FriendshipRequest.receiver_id == self.user_id,
                FriendshipRequest.status == 'pending'
            )

            blocked_users_stmt = select(friendship.c.friend_id).where(
                friendship.c.user_id == self.user_id,
                friendship.c.status == 'blocked'
            )

            # Build query to get users - ADD EXPLICIT JOIN with UserProfile
            query = (
                select(UserModel)
                .outerjoin(UserProfile, UserModel.id == UserProfile.user_id)  # Add this line
                .where(UserModel.id != self.user_id)
                .options(selectinload(UserModel.profile))
            )

            # Exclude friends, pending requests, and blocked users
            query = query.where(
                ~UserModel.id.in_(friends_stmt),
                ~UserModel.id.in_(sent_requests_stmt),
                ~UserModel.id.in_(received_requests_stmt),
                ~UserModel.id.in_(blocked_users_stmt)
            )

            # Apply search filter if provided
            if self.search:
                search_pattern = f"%{self.search}%"
                query = query.where(
                    or_(
                        UserModel.username.ilike(search_pattern),
                        UserModel.email.ilike(search_pattern),
                        UserProfile.first_name.ilike(search_pattern),  # This now works with the join
                        UserProfile.last_name.ilike(search_pattern)  # This now works with the join
                    )
                )

            query = query.order_by(UserModel.username).limit(self.limit)

            result = await self.db.execute(query)
            users = result.unique().scalars().all()  # Add .unique() to avoid duplicates

            # Format response
            formatted_users = []
            for user in users:
                # Check if there was any previous interaction
                previous_request_stmt = select(FriendshipRequest).where(
                    or_(
                        and_(
                            FriendshipRequest.sender_id == self.user_id,
                            FriendshipRequest.receiver_id == user.id
                        ),
                        and_(
                            FriendshipRequest.sender_id == user.id,
                            FriendshipRequest.receiver_id == self.user_id
                        )
                    )
                ).order_by(FriendshipRequest.created_at.desc())

                previous_request_result = await self.db.execute(previous_request_stmt)
                previous_request = previous_request_result.scalar_one_or_none()

                formatted_users.append({
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "profile": {
                        "first_name": user.profile.first_name if user.profile else None,
                        "last_name": user.profile.last_name if user.profile else None,
                        "profile_image_url": user.profile.profile_image_url if user.profile else None,
                        "country": user.profile.country if user.profile else None,
                        "city": user.profile.city if user.profile else None,
                        "bio": user.profile.bio if user.profile else None,
                    } if user.profile else None,
                    "native_language": user.native,
                    "is_premium": user.is_premium,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "relationship_status": self._get_relationship_status(
                        previous_request) if previous_request else "none",
                    "previous_request_id": previous_request.id if previous_request else None,
                    "previous_request_status": previous_request.status if previous_request else None,
                    "previous_request_date": previous_request.created_at.isoformat() if previous_request else None
                })

            return {
                "users": formatted_users,
                "total": len(formatted_users),
                "search": self.search,
                "limit": self.limit
            }

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error fetching users: {str(e)}"
            )



    # async def fetch_users(self) -> Dict:
    #     """
    #     Fetch users to show in frontend (excluding current user and friends)
    #     Returns users with friendship status
    #     """
    #     try:
    #         # Get current user's friends and pending requests
    #         friends_stmt = select(friendship.c.friend_id).where(
    #             friendship.c.user_id == self.user_id,
    #             friendship.c.status == 'accepted'
    #         )
    #
    #         sent_requests_stmt = select(FriendshipRequest.receiver_id).where(
    #             FriendshipRequest.sender_id == self.user_id,
    #             FriendshipRequest.status == 'pending'
    #         )
    #
    #         received_requests_stmt = select(FriendshipRequest.sender_id).where(
    #             FriendshipRequest.receiver_id == self.user_id,
    #             FriendshipRequest.status == 'pending'
    #         )
    #
    #         blocked_users_stmt = select(friendship.c.friend_id).where(
    #             friendship.c.user_id == self.user_id,
    #             friendship.c.status == 'blocked'
    #         )
    #
    #         # Build query to get users
    #         query = select(UserModel).where(
    #             UserModel.id != self.user_id
    #         ).options(
    #             selectinload(UserModel.profile)
    #         )
    #
    #         # Exclude friends, pending requests, and blocked users
    #         query = query.where(
    #             ~UserModel.id.in_(friends_stmt),
    #             ~UserModel.id.in_(sent_requests_stmt),
    #             ~UserModel.id.in_(received_requests_stmt),
    #             ~UserModel.id.in_(blocked_users_stmt)
    #         )
    #
    #         # Apply search filter if provided
    #         if self.search:
    #             search_pattern = f"%{self.search}%"
    #             query = query.where(
    #                 or_(
    #                     UserModel.username.ilike(search_pattern),
    #                     UserModel.email.ilike(search_pattern),
    #                     UserProfile.first_name.ilike(search_pattern),
    #                     UserProfile.last_name.ilike(search_pattern)
    #                 )
    #             )
    #
    #         query = query.order_by(UserModel.username).limit(self.limit)
    #
    #         result = await self.db.execute(query)
    #         users = result.scalars().all()
    #
    #         # Format response
    #         formatted_users = []
    #         for user in users:
    #             # Check if there was any previous interaction
    #             previous_request_stmt = select(FriendshipRequest).where(
    #                 or_(
    #                     and_(
    #                         FriendshipRequest.sender_id == self.user_id,
    #                         FriendshipRequest.receiver_id == user.id
    #                     ),
    #                     and_(
    #                         FriendshipRequest.sender_id == user.id,
    #                         FriendshipRequest.receiver_id == self.user_id
    #                     )
    #                 )
    #             ).order_by(FriendshipRequest.created_at.desc())
    #
    #             previous_request_result = await self.db.execute(previous_request_stmt)
    #             previous_request = previous_request_result.scalar_one_or_none()
    #
    #             formatted_users.append({
    #                 "id": user.id,
    #                 "username": user.username,
    #                 "email": user.email,
    #                 "profile": {
    #                     "first_name": user.profile.first_name if user.profile else None,
    #                     "last_name": user.profile.last_name if user.profile else None,
    #                     "profile_image_url": user.profile.profile_image_url if user.profile else None,
    #                     "country": user.profile.country if user.profile else None,
    #                     "city": user.profile.city if user.profile else None,
    #                     "bio": user.profile.bio if user.profile else None,
    #                 } if user.profile else None,
    #                 "native_language": user.native,
    #                 "is_premium": user.is_premium,
    #                 "created_at": user.created_at.isoformat() if user.created_at else None,
    #                 "relationship_status": self._get_relationship_status(
    #                     previous_request) if previous_request else "none",
    #                 "previous_request_id": previous_request.id if previous_request else None,
    #                 "previous_request_status": previous_request.status if previous_request else None,
    #                 "previous_request_date": previous_request.created_at.isoformat() if previous_request else None
    #             })
    #
    #
    #         return_data = {
    #             "users": formatted_users,
    #             "total": len(formatted_users),
    #             "search": self.search,
    #             "limit": self.limit
    #         }
    #
    #         print('return data is ....................', return_data)
    #
    #         # return {
    #         #     "users": formatted_users,
    #         #     "total": len(formatted_users),
    #         #     "search": self.search,
    #         #     "limit": self.limit
    #         # }
    #
    #         return return_data
    #
    #     except Exception as e:
    #         raise HTTPException(
    #             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #             detail=f"Error fetching users: {str(e)}"
    #         )

    def _get_relationship_status(self, request: FriendshipRequest) -> str:
        """Determine relationship status based on previous request"""
        if request.status == 'rejected':
            if request.sender_id == self.user_id:
                return "you_rejected"
            else:
                return "rejected_you"
        elif request.status == 'pending':
            if request.sender_id == self.user_id:
                return "request_sent"
            else:
                return "request_received"
        return "none"


class GetUserByIdRepository:
    def __init__(self, db: AsyncSession, user_id: int, getting_user_id: int):
        self.db = db
        self.user_id = user_id
        self.getting_user_id = getting_user_id

    async def get_user_by_id(self) -> Dict:
        """Get user by ID with relationship info"""
        try:
            # Get the target user
            stmt = select(UserModel).where(
                UserModel.id == self.getting_user_id
            ).options(
                selectinload(UserModel.profile)
            )

            result = await self.db.execute(stmt)
            user = result.scalar_one_or_none()

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )

            # Check relationship
            friendship_stmt = select(friendship).where(
                or_(
                    and_(
                        friendship.c.user_id == self.user_id,
                        friendship.c.friend_id == self.getting_user_id
                    ),
                    and_(
                        friendship.c.user_id == self.getting_user_id,
                        friendship.c.friend_id == self.user_id
                    )
                )
            )

            friendship_result = await self.db.execute(friendship_stmt)
            friendship_data = friendship_result.first()

            # Check pending requests
            pending_request_stmt = select(FriendshipRequest).where(
                or_(
                    and_(
                        FriendshipRequest.sender_id == self.user_id,
                        FriendshipRequest.receiver_id == self.getting_user_id,
                        FriendshipRequest.status == 'pending'
                    ),
                    and_(
                        FriendshipRequest.sender_id == self.getting_user_id,
                        FriendshipRequest.receiver_id == self.user_id,
                        FriendshipRequest.status == 'pending'
                    )
                )
            )

            pending_request_result = await self.db.execute(pending_request_stmt)
            pending_request = pending_request_result.scalar_one_or_none()

            relationship_status = "none"
            if friendship_data:
                if friendship_data.status == 'accepted':
                    relationship_status = "friends"
                elif friendship_data.status == 'blocked':
                    relationship_status = "blocked"
            elif pending_request:
                if pending_request.sender_id == self.user_id:
                    relationship_status = "request_sent"
                else:
                    relationship_status = "request_received"

            # Get common friends count
            common_friends_stmt = select(func.count(friendship.c.friend_id)).where(
                friendship.c.user_id == self.user_id,
                friendship.c.status == 'accepted',
                friendship.c.friend_id.in_(
                    select(friendship.c.friend_id).where(
                        friendship.c.user_id == self.getting_user_id,
                        friendship.c.status == 'accepted'
                    )
                )
            )

            common_friends_result = await self.db.execute(common_friends_stmt)
            common_friends_count = common_friends_result.scalar() or 0

            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "profile": {
                    "first_name": user.profile.first_name if user.profile else None,
                    "last_name": user.profile.last_name if user.profile else None,
                    "profile_image_url": user.profile.profile_image_url if user.profile else None,
                    "country": user.profile.country if user.profile else None,
                    "city": user.profile.city if user.profile else None,
                    "bio": user.profile.bio if user.profile else None,
                    "gender": user.profile.gender if user.profile else None,
                    "date_of_birth": user.profile.date_of_birth.isoformat() if user.profile and user.profile.date_of_birth else None,
                } if user.profile else None,
                "native_language": user.native,
                "is_premium": user.is_premium,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "relationship_status": relationship_status,
                "common_friends_count": common_friends_count,
                "can_send_request": relationship_status in ["none", "rejected_you", "you_rejected"],
                "is_online": False,  # Will be populated by Express WebSocket
                "last_seen": None  # Will be populated by Express WebSocket
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error fetching user: {str(e)}"
            )


class GetFriendRequestsRepository:
    def __init__(self, db: AsyncSession, user_id: int):
        self.db = db
        self.user_id = user_id

    async def get_friend_requests(self) -> Dict:
        """
        Get all pending friend requests for the current user
        Returns sent and received requests separately
        """
        try:
            # Get received pending requests (people who want to be friends with you)
            received_stmt = (
                select(FriendshipRequest)
                .join(UserModel, FriendshipRequest.sender_id == UserModel.id)
                .outerjoin(UserProfile, UserModel.id == UserProfile.user_id)
                .where(
                    and_(
                        FriendshipRequest.receiver_id == self.user_id,
                        FriendshipRequest.status == 'pending'
                    )
                )
                .order_by(desc(FriendshipRequest.created_at))
                .options(
                    selectinload(FriendshipRequest.sender).selectinload(UserModel.profile)
                )
            )

            # Get sent pending requests (requests you sent to others)
            sent_stmt = (
                select(FriendshipRequest)
                .join(UserModel, FriendshipRequest.receiver_id == UserModel.id)
                .outerjoin(UserProfile, UserModel.id == UserProfile.user_id)
                .where(
                    and_(
                        FriendshipRequest.sender_id == self.user_id,
                        FriendshipRequest.status == 'pending'
                    )
                )
                .order_by(desc(FriendshipRequest.created_at))
                .options(
                    selectinload(FriendshipRequest.receiver).selectinload(UserModel.profile)
                )
            )

            # Execute both queries
            received_result = await self.db.execute(received_stmt)
            received_requests = received_result.scalars().all()

            sent_result = await self.db.execute(sent_stmt)
            sent_requests = sent_result.scalars().all()

            # Format received requests
            formatted_received = []
            for req in received_requests:
                formatted_received.append({
                    "id": req.id,
                    "sender": self._format_user(req.sender),
                    "status": req.status,
                    "created_at": req.created_at.isoformat() if req.created_at else None,
                    "updated_at": req.updated_at.isoformat() if req.updated_at else None
                })

            # Format sent requests
            formatted_sent = []
            for req in sent_requests:
                formatted_sent.append({
                    "id": req.id,
                    "receiver": self._format_user(req.receiver),
                    "status": req.status,
                    "created_at": req.created_at.isoformat() if req.created_at else None,
                    "updated_at": req.updated_at.isoformat() if req.updated_at else None
                })

            return {
                "received_requests": formatted_received,
                "sent_requests": formatted_sent,
                "counts": {
                    "received": len(formatted_received),
                    "sent": len(formatted_sent),
                    "total": len(formatted_received) + len(formatted_sent)
                }
            }

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error fetching friend requests: {str(e)}"
            )

    def _format_user(self, user: UserModel) -> Dict:
        """Format user information"""
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "profile": {
                "first_name": user.profile.first_name if user.profile else None,
                "last_name": user.profile.last_name if user.profile else None,
                "profile_image_url": user.profile.profile_image_url if user.profile else None,
                "country": user.profile.country if user.profile else None,
                "city": user.profile.city if user.profile else None,
                "bio": user.profile.bio if user.profile else None,
            } if user.profile else None,
            "native_language": user.native,
            "is_premium": user.is_premium,
            "created_at": user.created_at.isoformat() if user.created_at else None
        }