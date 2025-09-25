"""create_table_category

Revision ID: d537b0e9cfa6
Revises: 6bfd788bcb85
Create Date: 2025-09-25 20:11:01.447104

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd537b0e9cfa6'
down_revision: Union[str, None] = '6bfd788bcb85'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('categories',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('name', sa.String(length=50), nullable=False),
                    sa.Column('description', sa.String(length=200), nullable=True),
                    sa.PrimaryKeyConstraint('id'),
                    sa.UniqueConstraint('name')
                    )

    op.create_table('word_categories',
                    sa.Column('word_id', sa.Integer(), nullable=False),
                    sa.Column('category_id', sa.Integer(), nullable=False),
                    sa.ForeignKeyConstraint(['category_id'], ['categories.id'], ),
                    sa.ForeignKeyConstraint(['word_id'], ['words.id'], ),
                    sa.PrimaryKeyConstraint('word_id', 'category_id')
                    )

    op.create_index('ix_word_categories_word_id', 'word_categories', ['word_id'])
    op.create_index('ix_word_categories_category_id', 'word_categories', ['category_id'])
    op.create_index('ix_words_text', 'words', ['text'])
    op.create_index('ix_categories_name', 'categories', ['name'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('word_categories')
    op.drop_table('categories')
