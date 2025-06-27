"""Create user_saved_words

Revision ID: 650c2ae18532
Revises: ec24534763a6
Create Date: 2025-06-27 14:14:32.321864

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '650c2ae18532'
down_revision: Union[str, None] = 'ec24534763a6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Create 'user_saved_words' table."""

    op.create_table(
        'user_saved_words',
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), primary_key=True),
        sa.Column('word_id', sa.Integer(), sa.ForeignKey('words.id'), primary_key=True),
        sa.Column('learned', sa.Boolean(), default=False),
        sa.Column('starred', sa.Boolean(), default=False),
        sa.Column('last_reviewed', sa.DateTime(timezone=True))
    )

    # Add indexes for fast lookups
    op.create_index('idx_user_saved_words_user', 'user_saved_words', ['user_id'])
    op.create_index('idx_user_saved_words_word', 'user_saved_words', ['word_id'])


def downgrade() -> None:
    """Drop 'user_saved_words' table."""

    op.drop_index('idx_user_saved_words_user', table_name='user_saved_words')
    op.drop_index('idx_user_saved_words_word', table_name='user_saved_words')
    op.drop_table('user_saved_words')