"""create table sentence_words model

Revision ID: 7980ef9a6d90
Revises: 0332e9abaef3
Create Date: 2025-07-05 10:57:46.028748

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7980ef9a6d90'
down_revision: Union[str, None] = '0332e9abaef3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('sentence_words',
        sa.Column('sentence_id', sa.Integer(), nullable=False),
        sa.Column('word_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['sentence_id'], ['sentences.id']),
        sa.ForeignKeyConstraint(['word_id'], ['words.id']),
        sa.PrimaryKeyConstraint('sentence_id', 'word_id'),
        sa.Index('ix_sentence_words_word', 'word_id'),
        sa.Index('ix_sentence_words_sentence', 'sentence_id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('sentence_words')