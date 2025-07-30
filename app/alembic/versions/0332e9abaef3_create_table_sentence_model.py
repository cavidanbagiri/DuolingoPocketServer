"""create table sentence model

Revision ID: 0332e9abaef3
Revises: 9c897c187767
Create Date: 2025-07-05 10:57:04.508030

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0332e9abaef3'
down_revision: Union[str, None] = '9c897c187767'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('sentences',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('text', sa.String(length=500), nullable=False),
        sa.Column('language_code', sa.String(length=2), nullable=False),
        sa.ForeignKeyConstraint(['language_code'], ['languages.code']),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_sentences_language', 'language_code')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('sentences')
