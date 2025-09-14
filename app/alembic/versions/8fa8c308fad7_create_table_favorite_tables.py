"""create_table_favorite_tables

Revision ID: 8fa8c308fad7
Revises: b2464ece5b02
Create Date: 2025-09-14 19:19:56.736085

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8fa8c308fad7'
down_revision: Union[str, None] = 'b2464ece5b02'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create favorite_categories table
    op.create_table(
        'favorite_categories',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    )
    op.create_index(op.f('ix_favorite_categories_id'), 'favorite_categories', ['id'], unique=False)
    op.create_index(op.f('ix_favorite_categories_user_id'), 'favorite_categories', ['user_id'], unique=False)

    # Create favorite_words table
    op.create_table(
        'favorite_words',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('category_id', sa.Integer(), nullable=True),
        sa.Column('from_lang', sa.String(length=50), nullable=False),
        sa.Column('to_lang', sa.String(length=50), nullable=False),
        sa.Column('original_text', sa.String(), nullable=False),
        sa.Column('translated_text', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['category_id'], ['favorite_categories.id'], ondelete='SET NULL'),
    )
    op.create_index(op.f('ix_favorite_words_id'), 'favorite_words', ['id'], unique=False)
    op.create_index(op.f('ix_favorite_words_user_id'), 'favorite_words', ['user_id'], unique=False)
    op.create_index(op.f('ix_favorite_words_category_id'), 'favorite_words', ['category_id'], unique=False)

    # Create default_categories table
    op.create_table(
        'default_categories',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    )
    op.create_index(op.f('ix_default_categories_id'), 'default_categories', ['id'], unique=False)
    op.create_index(op.f('ix_default_categories_user_id'), 'default_categories', ['user_id'], unique=False)

def downgrade():
    # Drop tables in reverse order
    op.drop_index(op.f('ix_default_categories_user_id'), table_name='default_categories')
    op.drop_index(op.f('ix_default_categories_id'), table_name='default_categories')
    op.drop_table('default_categories')

    op.drop_index(op.f('ix_favorite_words_category_id'), table_name='favorite_words')
    op.drop_index(op.f('ix_favorite_words_user_id'), table_name='favorite_words')
    op.drop_index(op.f('ix_favorite_words_id'), table_name='favorite_words')
    op.drop_table('favorite_words')

    op.drop_index(op.f('ix_favorite_categories_user_id'), table_name='favorite_categories')
    op.drop_index(op.f('ix_favorite_categories_id'), table_name='favorite_categories')
    op.drop_table('favorite_categories')
