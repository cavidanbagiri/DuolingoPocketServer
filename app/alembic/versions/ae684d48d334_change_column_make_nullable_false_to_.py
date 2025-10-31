"""change_column_make_nullable_false_to_password_field_users

Revision ID: ae684d48d334
Revises: 7e9f77227da7
Create Date: 2025-10-31 18:20:13.682814

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ae684d48d334'
down_revision: Union[str, None] = '7e9f77227da7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column(
        'users',
        'password',
        nullable=True,
        existing_type=sa.String(length=255)  # Specify the existing type if needed
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column(
        'users',
        'password',
        nullable=False,
        existing_type=sa.String(length=255)
    )
