"""create table learned_words model

Revision ID: 43744439e4d3
Revises: bd348b3e15c4
Create Date: 2025-07-05 10:59:06.380081

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '43744439e4d3'
down_revision: Union[str, None] = 'bd348b3e15c4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('learned_words',
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('word_id', sa.Integer(), nullable=False),
        sa.Column('target_language', sa.String(length=2), nullable=False),
        sa.Column('last_practiced', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('strength', sa.Integer(), server_default=sa.text('0')),
        sa.ForeignKeyConstraint(['word_id'], ['words.id']),
        sa.PrimaryKeyConstraint('user_id', 'word_id', 'target_language'),
        sa.Index('ix_learned_words_user_lang', 'user_id', 'target_language'),
        sa.Index('ix_learned_words_strength', 'user_id', 'strength'),
        sa.Index('ix_learned_words_practice', 'user_id', 'last_practiced')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('learned_words')