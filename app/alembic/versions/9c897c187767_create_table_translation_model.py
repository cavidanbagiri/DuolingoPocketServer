"""create table translation model

Revision ID: 9c897c187767
Revises: e099ded75648
Create Date: 2025-07-05 10:56:26.903916

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9c897c187767'
down_revision: Union[str, None] = 'e099ded75648'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('translations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('source_word_id', sa.Integer(), nullable=False),
        sa.Column('target_language_code', sa.String(length=2), nullable=False),
        sa.Column('translated_text', sa.String(length=100), nullable=False),
        sa.ForeignKeyConstraint(['source_word_id'], ['words.id']),
        sa.ForeignKeyConstraint(['target_language_code'], ['languages.code']),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_translations_source_target', 'source_word_id', 'target_language_code', unique=True),
        sa.Index('ix_translations_text', 'translated_text')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('translations', if_exists=True)
