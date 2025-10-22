"""add_column_update_at_to_user_words

Revision ID: 7e9f77227da7
Revises: d537b0e9cfa6
Create Date: 2025-10-22 15:01:53.481328

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7e9f77227da7'
down_revision: Union[str, None] = 'd537b0e9cfa6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('user_words',
                  sa.Column('updated_at',
                            sa.DateTime(timezone=True),
                            server_default=sa.func.now(),
                            nullable=True)
                  )

    # Set initial value for existing records
    op.execute("UPDATE user_words SET updated_at = created_at")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('user_words', 'updated_at')
