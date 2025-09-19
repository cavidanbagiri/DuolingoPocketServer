"""add_column added_at_to_favorite_words 

Revision ID: ae0471e55cf7
Revises: 8fa8c308fad7
Create Date: 2025-09-19 11:32:39.383535

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ae0471e55cf7'
down_revision: Union[str, None] = '8fa8c308fad7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column('favorite_words',
                  sa.Column('added_at', sa.DateTime(), nullable=True)
                  )

    # Set default value for existing records
    op.execute("UPDATE favorite_words SET added_at = NOW() WHERE added_at IS NULL")

    # Make column not nullable after setting defaults
    op.alter_column('favorite_words', 'added_at', nullable=False)


def downgrade():
    op.drop_column('favorite_words', 'added_at')


# def upgrade() -> None:
#     """Upgrade schema."""
#     pass
#
#
# def downgrade() -> None:
#     """Downgrade schema."""
#     pass
