"""add_column_color_icon_to_favorite_model

Revision ID: 69e04a70c3cf
Revises: ae0471e55cf7
Create Date: 2025-09-21 11:29:32.973068

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '69e04a70c3cf'
down_revision: Union[str, None] = 'ae0471e55cf7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('favorite_categories',
                  sa.Column('icon', sa.String, nullable=True))


    op.add_column('favorite_categories',
                  sa.Column('color', sa.String, nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('favorite_categories', 'icon')
    op.drop_column('favorite_categories', 'color')
