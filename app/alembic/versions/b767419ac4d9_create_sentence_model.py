"""Create sentence model

Revision ID: b767419ac4d9
Revises: 650c2ae18532
Create Date: 2025-06-27 14:22:18.140145

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b767419ac4d9'
down_revision: Union[str, None] = '650c2ae18532'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# alembic/versions/zzzz_create_sentence_hints_table.py

from alembic import op
import sqlalchemy as sa


def upgrade() -> None:
    """Create 'sentence_hints' table."""

    op.create_table(
        'sentence_hints',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True),
        sa.Column('sentence', sa.String(), nullable=False),
        sa.Column('translation', sa.String(), nullable=False),
        sa.Column('word_id', sa.Integer(), sa.ForeignKey('words.id'), nullable=False)
    )


    # Relationship to words
    op.create_index('idx_sentence_hints_word', 'sentence_hints', ['word_id'])


def downgrade() -> None:
    """Drop 'sentence_hints' table."""

    op.drop_index('idx_sentence_hints_word', table_name='sentence_hints')
    op.drop_table('sentence_hints')