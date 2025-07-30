"""create table word model

Revision ID: e099ded75648
Revises: 75f8f2bba61a
Create Date: 2025-07-05 10:55:37.618661

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e099ded75648'
down_revision: Union[str, None] = '75f8f2bba61a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('words',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('text', sa.String(length=100), nullable=False),
        sa.Column('language_code', sa.String(length=2), nullable=False),
        # sa.Column('pos', sa.String(length=20), nullable=True),
        sa.Column('level', sa.String(length=2), nullable=True),
        sa.Column('frequency_rank', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['language_code'], ['languages.code']),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_words_text_lang', 'text', 'language_code'),
        sa.Index('ix_words_lang_freq', 'language_code', 'frequency_rank'),
        sa.Index('ix_words_lang_level', 'language_code', 'level')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('words', if_exists=True)
