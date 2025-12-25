"""create_table_notes

Revision ID: 6e29c3d9b3ae
Revises: 6326cc5804b7
Create Date: 2025-12-24 19:01:05.883249

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '6e29c3d9b3ae'
down_revision: Union[str, None] = '6326cc5804b7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Create notes table
    op.create_table('notes',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('note_name', sa.String(length=200), nullable=False),
                    sa.Column('user_id', sa.Integer(), nullable=False),
                    sa.Column('target_lang', sa.String(length=10), nullable=True),
                    sa.Column('note_type', sa.String(length=20), nullable=False),
                    sa.Column('content', sa.Text(), nullable=False),
                    sa.Column('tags', postgresql.ARRAY(sa.String(length=50)), nullable=True),
                    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
                    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
                    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
                    sa.PrimaryKeyConstraint('id')
                    )

    # Create indexes
    op.create_index('idx_notes_user_id', 'notes', ['user_id'])
    op.create_index('idx_notes_note_name', 'notes', ['note_name'])
    op.create_index('idx_notes_updated_at', 'notes', ['updated_at'])

    # Composite index for user + created_at (for chronological listing)
    op.create_index('idx_notes_user_created', 'notes', ['user_id', 'created_at'])

    # Partial index for user + target_lang (only non-null values)
    op.create_index(
        'idx_notes_user_target_lang_partial',
        'notes',
        ['user_id', 'target_lang'],
        postgresql_where=sa.text('target_lang IS NOT NULL')
    )

    # GIN index for tags array operations
    op.create_index(
        'gin_idx_notes_tags',
        'notes',
        ['tags'],
        postgresql_using='gin'
    )


def downgrade():
    # Drop indexes first
    op.drop_index('gin_idx_notes_tags', table_name='notes')
    op.drop_index('idx_notes_user_target_lang_partial', table_name='notes')
    op.drop_index('idx_notes_user_created', table_name='notes')
    op.drop_index('idx_notes_updated_at', table_name='notes')
    op.drop_index('idx_notes_note_name', table_name='notes')
    op.drop_index('idx_notes_user_id', table_name='notes')

    # Drop table
    op.drop_table('notes')