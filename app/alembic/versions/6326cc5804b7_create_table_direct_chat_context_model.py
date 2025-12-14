"""create_table_direct_chat_context_model

Revision ID: 6326cc5804b7
Revises: a4d1a1181a49
Create Date: 2025-12-14 14:31:42.462677

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6326cc5804b7'
down_revision: Union[str, None] = 'a4d1a1181a49'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# def upgrade() -> None:
#     """Upgrade schema."""
#     pass
#
#
# def downgrade() -> None:
#     """Downgrade schema."""
#     pass


# alembic/versions/xxxxx_add_direct_chat_contexts_table.py

def upgrade():
    op.create_table('direct_chat_contexts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('topic', sa.String(length=255), nullable=True),
        sa.Column('messages', sa.Text(), nullable=False),
        sa.Column('context_hash', sa.String(length=64), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('context_hash')
    )
    op.create_index(op.f('ix_direct_chat_contexts_context_hash'), 'direct_chat_contexts', ['context_hash'], unique=False)
    op.create_index(op.f('ix_direct_chat_contexts_user_id'), 'direct_chat_contexts', ['user_id'], unique=False)
    op.create_index(op.f('ix_direct_chat_contexts_topic'), 'direct_chat_contexts', ['topic'], unique=False)

def downgrade():
    op.drop_index(op.f('ix_direct_chat_contexts_topic'), table_name='direct_chat_contexts')
    op.drop_index(op.f('ix_direct_chat_contexts_user_id'), table_name='direct_chat_contexts')
    op.drop_index(op.f('ix_direct_chat_contexts_context_hash'), table_name='direct_chat_contexts')
    op.drop_table('direct_chat_contexts')