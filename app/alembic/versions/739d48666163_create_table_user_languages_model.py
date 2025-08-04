"""create table user languages model

Revision ID: 739d48666163
Revises: 299fbc5d6179
Create Date: 2025-08-04 11:06:04.704316

"""
from datetime import datetime
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '739d48666163'
down_revision: Union[str, None] = '299fbc5d6179'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'user_languages',
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), primary_key=True),
        sa.Column('target_language_code', sa.String(length=2), sa.ForeignKey('languages.code')),
        sa.Column('level', sa.String(length=2), nullable=True, server_default='A1'),
        sa.Column('updated_at', sa.DateTime(), default=datetime.utcnow)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('user_languages')
