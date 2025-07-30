"""create_table_word_meaning_model

Revision ID: 299fbc5d6179
Revises: 43744439e4d3
Create Date: 2025-07-06 14:36:17.581457

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '299fbc5d6179'
down_revision: Union[str, None] = '43744439e4d3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create word_meanings table
    op.create_table('word_meanings',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('word_id', sa.Integer(), nullable=False),
                    sa.Column('pos', sa.String(length=50), nullable=False),
                    #sa.Column('definition', sa.String(length=500), nullable=True),
                    sa.Column('example', sa.String(length=500), nullable=True),
                    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
                    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), onupdate=sa.text('now()'),
                              nullable=False),
                    sa.ForeignKeyConstraint(['word_id'], ['words.id'], ondelete='CASCADE'),
                    sa.PrimaryKeyConstraint('id')
                    )

    # Create meaning_sentence_links junction table
    op.create_table('meaning_sentence_links',
                    sa.Column('meaning_id', sa.Integer(), nullable=False),
                    sa.Column('sentence_id', sa.Integer(), nullable=False),
                    sa.ForeignKeyConstraint(['meaning_id'], ['word_meanings.id'], ondelete='CASCADE'),
                    sa.ForeignKeyConstraint(['sentence_id'], ['sentences.id'], ondelete='CASCADE'),
                    sa.PrimaryKeyConstraint('meaning_id', 'sentence_id')
                    )

    # Add indexes for performance
    op.create_index('ix_word_meanings_word_id', 'word_meanings', ['word_id'])
    op.create_index('ix_word_meanings_pos', 'word_meanings', ['pos'])
    op.create_index('ix_meaning_sentence_links_meaning', 'meaning_sentence_links', ['meaning_id'])
    op.create_index('ix_meaning_sentence_links_sentence', 'meaning_sentence_links', ['sentence_id'])


def downgrade() -> None:
    """Downgrade schema."""
    # Drop junction table first (due to foreign keys)
    op.drop_table('meaning_sentence_links')

    # Then drop word_meanings table
    op.drop_table('word_meanings')
