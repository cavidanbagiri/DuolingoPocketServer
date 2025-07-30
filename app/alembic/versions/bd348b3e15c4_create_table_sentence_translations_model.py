"""create table sentence_translations model

Revision ID: bd348b3e15c4
Revises: 7980ef9a6d90
Create Date: 2025-07-05 10:58:28.401831

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bd348b3e15c4'
down_revision: Union[str, None] = '7980ef9a6d90'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('sentence_translations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('source_sentence_id', sa.Integer(), nullable=False),
        sa.Column('language_code', sa.String(length=2), nullable=False),
        sa.Column('translated_text', sa.String(length=500), nullable=False),
        sa.ForeignKeyConstraint(['source_sentence_id'], ['sentences.id']),
        sa.ForeignKeyConstraint(['language_code'], ['languages.code']),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_sentence_trans_source_lang', 'source_sentence_id', 'language_code', unique=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('sentence_translations')