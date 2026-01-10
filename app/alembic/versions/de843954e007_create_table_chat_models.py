"""create_table_chat_models

Revision ID: de843954e007
Revises: 6e29c3d9b3ae
Create Date: 2026-01-07 21:19:55.200646

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'de843954e007'
down_revision: Union[str, None] = '6e29c3d9b3ae'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Create user_profiles table
    op.create_table('user_profiles',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('user_id', sa.Integer(), nullable=False),
                    sa.Column('first_name', sa.String(length=100), nullable=True),
                    sa.Column('middle_name', sa.String(length=100), nullable=True),
                    sa.Column('last_name', sa.String(length=100), nullable=True),
                    sa.Column('date_of_birth', sa.Date(), nullable=True),
                    sa.Column('age', sa.Integer(), nullable=True),
                    sa.Column('country', sa.String(length=100), nullable=True),
                    sa.Column('city', sa.String(length=100), nullable=True),
                    sa.Column('gender', sa.String(length=20), nullable=True),
                    sa.Column('bio', sa.Text(), nullable=True),
                    sa.Column('profile_image_url', sa.String(length=500), nullable=True),
                    sa.Column('cover_image_url', sa.String(length=500), nullable=True),
                    sa.Column('phone_number', sa.String(length=20), nullable=True),
                    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
                    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
                    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
                    sa.PrimaryKeyConstraint('id'),
                    sa.UniqueConstraint('user_id')
                    )
    op.create_index(op.f('ix_user_profiles_id'), 'user_profiles', ['id'], unique=False)

    # Add check constraint for gender
    op.create_check_constraint(
        'check_gender_values',
        'user_profiles',
        "gender IN ('male', 'female', 'other', 'prefer_not_to_say')"
    )

    # Create friendships table (association table)
    op.create_table('friendships',
                    sa.Column('user_id', sa.Integer(), nullable=False),
                    sa.Column('friend_id', sa.Integer(), nullable=False),
                    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
                    sa.Column('status', sa.String(length=20), nullable=True),
                    sa.ForeignKeyConstraint(['friend_id'], ['users.id'], ondelete='CASCADE'),
                    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
                    sa.PrimaryKeyConstraint('user_id', 'friend_id')
                    )

    # Create friendship_requests table
    op.create_table('friendship_requests',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('sender_id', sa.Integer(), nullable=True),
                    sa.Column('receiver_id', sa.Integer(), nullable=True),
                    sa.Column('status', sa.String(length=20), nullable=True),
                    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
                    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
                    sa.ForeignKeyConstraint(['receiver_id'], ['users.id'], ondelete='CASCADE'),
                    sa.ForeignKeyConstraint(['sender_id'], ['users.id'], ondelete='CASCADE'),
                    sa.PrimaryKeyConstraint('id')
                    )
    op.create_index(op.f('ix_friendship_requests_id'), 'friendship_requests', ['id'], unique=False)

    # Add check constraint for friendship_requests status
    op.create_check_constraint(
        'check_friend_request_status',
        'friendship_requests',
        "status IN ('pending', 'accepted', 'rejected')"
    )

    # Create conversations table
    op.create_table('conversations',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('is_group', sa.Boolean(), nullable=True),
                    sa.Column('group_name', sa.String(length=200), nullable=True),
                    sa.Column('group_image_url', sa.String(length=500), nullable=True),
                    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
                    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
                    sa.PrimaryKeyConstraint('id')
                    )
    op.create_index(op.f('ix_conversations_id'), 'conversations', ['id'], unique=False)

    # Create messages table
    op.create_table('messages',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('conversation_id', sa.Integer(), nullable=True),
                    sa.Column('sender_id', sa.Integer(), nullable=True),
                    sa.Column('content', sa.Text(), nullable=True),
                    sa.Column('message_type', sa.String(length=20), nullable=True),
                    sa.Column('media_url', sa.String(length=500), nullable=True),
                    sa.Column('media_thumbnail_url', sa.String(length=500), nullable=True),
                    sa.Column('file_size', sa.Integer(), nullable=True),
                    sa.Column('is_edited', sa.Boolean(), nullable=True),
                    sa.Column('is_deleted', sa.Boolean(), nullable=True),
                    sa.Column('reply_to_message_id', sa.Integer(), nullable=True),
                    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
                    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
                    sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE'),
                    sa.ForeignKeyConstraint(['reply_to_message_id'], ['messages.id'], ),
                    sa.ForeignKeyConstraint(['sender_id'], ['users.id'], ondelete='CASCADE'),
                    sa.PrimaryKeyConstraint('id')
                    )
    op.create_index(op.f('ix_messages_id'), 'messages', ['id'], unique=False)

    # Add check constraint for messages message_type
    op.create_check_constraint(
        'check_message_type',
        'messages',
        "message_type IN ('text', 'image', 'video', 'audio', 'file')"
    )

    # Create conversation_participants table
    op.create_table('conversation_participants',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('conversation_id', sa.Integer(), nullable=True),
                    sa.Column('user_id', sa.Integer(), nullable=True),
                    sa.Column('joined_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
                    sa.Column('left_at', sa.DateTime(timezone=True), nullable=True),
                    sa.Column('is_admin', sa.Boolean(), nullable=True),
                    sa.Column('is_muted', sa.Boolean(), nullable=True),
                    sa.Column('last_read_message_id', sa.Integer(), nullable=True),
                    sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE'),
                    sa.ForeignKeyConstraint(['last_read_message_id'], ['messages.id'], ),
                    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
                    sa.PrimaryKeyConstraint('id')
                    )
    op.create_index(op.f('ix_conversation_participants_id'), 'conversation_participants', ['id'], unique=False)

    # Add unique constraint for conversation_participants
    op.create_unique_constraint(
        'uq_conversation_participant',
        'conversation_participants',
        ['conversation_id', 'user_id']
    )

    # Create message_status table
    op.create_table('message_status',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('message_id', sa.Integer(), nullable=True),
                    sa.Column('user_id', sa.Integer(), nullable=True),
                    sa.Column('status', sa.String(length=20), nullable=True),
                    sa.Column('read_at', sa.DateTime(timezone=True), nullable=True),
                    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
                    sa.ForeignKeyConstraint(['message_id'], ['messages.id'], ondelete='CASCADE'),
                    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
                    sa.PrimaryKeyConstraint('id')
                    )
    op.create_index(op.f('ix_message_status_id'), 'message_status', ['id'], unique=False)

    # Add check constraint for message_status
    op.create_check_constraint(
        'check_delivery_status',
        'message_status',
        "status IN ('sent', 'delivered', 'read')"
    )

    # Add unique constraint for message_status
    op.create_unique_constraint(
        'uq_message_status',
        'message_status',
        ['message_id', 'user_id']
    )

    # Create typing_indicators table
    op.create_table('typing_indicators',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('conversation_id', sa.Integer(), nullable=True),
                    sa.Column('user_id', sa.Integer(), nullable=True),
                    sa.Column('is_typing', sa.Boolean(), nullable=True),
                    sa.Column('last_updated', sa.DateTime(timezone=True), server_default=sa.text('now()'),
                              nullable=True),
                    sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE'),
                    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
                    sa.PrimaryKeyConstraint('id')
                    )
    op.create_index(op.f('ix_typing_indicators_id'), 'typing_indicators', ['id'], unique=False)

    # Add unique constraint for typing_indicators
    op.create_unique_constraint(
        'uq_typing_indicator',
        'typing_indicators',
        ['conversation_id', 'user_id']
    )


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('typing_indicators')
    op.drop_table('message_status')
    op.drop_table('conversation_participants')
    op.drop_table('messages')
    op.drop_table('conversations')
    op.drop_table('friendship_requests')
    op.drop_table('friendships')
    op.drop_table('user_profiles')