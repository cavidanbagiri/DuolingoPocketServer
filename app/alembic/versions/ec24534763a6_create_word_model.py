"""Create word model

Revision ID: ec24534763a6
Revises: 4f72430125d2
Create Date: 2025-06-27 12:56:54.486841

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import func

# revision identifiers, used by Alembic.
revision: str = 'ec24534763a6'
down_revision: Union[str, None] = '4f72430125d2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create 'words' table."""

    op.create_table(
        'words',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True),
        sa.Column('from_lang', sa.String(), nullable=True),
        sa.Column('to_lang', sa.String(), nullable=True),
        sa.Column('word', sa.String(), nullable=False),
        sa.Column('part_of_speech', sa.String(), nullable=False, default='other'),
        sa.Column('translation', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=func.now())
    )



def downgrade() -> None:
    """Drop 'words' table."""

    op.drop_table('words')