"""change_column_to_word_meaning_example_with_definition

Revision ID: 6bfd788bcb85
Revises: 69e04a70c3cf
Create Date: 2025-09-25 13:17:41.526066

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6bfd788bcb85'
down_revision: Union[str, None] = '69e04a70c3cf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_column('word_meanings', 'example')
    op.add_column('word_meanings', sa.Column('definition', sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('word_meanings', 'definition')
    op.add_column('word_meanings', sa.Column('example', sa.Text(), nullable=True))
