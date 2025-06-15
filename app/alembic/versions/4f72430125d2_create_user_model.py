"""create user model

Revision ID: 4f72430125d2
Revises: 
Create Date: 2025-06-11 23:44:56.387985

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4f72430125d2'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('users',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('username', sa.String(), nullable=True),
                    sa.Column('email', sa.String(length=100), nullable=False),
                    sa.Column('password', sa.String(length=100), nullable=False),
                    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text("now()")),
                    sa.Column('is_premium', sa.Boolean(), nullable=False, default=False),
                    sa.Column('role', sa.String(), nullable=False, default='user'),
                    sa.PrimaryKeyConstraint('id'),
                    sa.UniqueConstraint('email'),
                    sa.UniqueConstraint('username')
                    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)

    op.create_table('tokens',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('tokens', sa.String(), nullable=False),
                    sa.Column('user_id', sa.Integer(), nullable=False),
                    sa.ForeignKeyConstraint(['user_id'], ['users.id']),
                    sa.PrimaryKeyConstraint('id')
                    )
    op.create_index(op.f('ix_tokens_tokens'), 'tokens', ['tokens'], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_tokens_tokens'), table_name='tokens')
    op.drop_table('tokens')

    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
