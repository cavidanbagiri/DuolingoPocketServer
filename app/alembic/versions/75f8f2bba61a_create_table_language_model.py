"""create table language model

Revision ID: 75f8f2bba61a
Revises: 6ddbc995ae72
Create Date: 2025-07-05 10:54:56.692766

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '75f8f2bba61a'
down_revision: Union[str, None] = '6ddbc995ae72'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('languages',
                    sa.Column('code', sa.String(length=2), nullable=False),
                    sa.Column('name', sa.String(length=50), nullable=True),
                    sa.PrimaryKeyConstraint('code'),
                    sa.Index('ix_languages_code', 'code', unique=True)
                    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('languages')
