"""create_index_for_user_words

Revision ID: 3e593903ea75
Revises: de843954e007
Create Date: 2026-01-27 21:26:39.054294

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3e593903ea75'
down_revision: Union[str, None] = 'de843954e007'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # For CONCURRENTLY indexes, we need to run outside transaction
    # Method 1: Use op.get_bind() with autocommit
    connection = op.get_bind()

    # Set autocommit mode for CONCURRENTLY operations
    connection.execute(sa.text("COMMIT"))  # End current transaction

    # Now create indexes with CONCURRENTLY
    print("Creating index: idx_user_words_user_id...")
    connection.execute(sa.text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_words_user_id 
        ON user_words(user_id)
    """))

    print("Creating index: idx_user_words_user_word...")
    connection.execute(sa.text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_words_user_word 
        ON user_words(user_id, word_id)
    """))

    print("Creating index: idx_user_words_is_learned...")
    connection.execute(sa.text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_words_is_learned 
        ON user_words(user_id) 
        WHERE is_learned = true
    """))

    print("Creating index: idx_user_words_is_starred...")
    connection.execute(sa.text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_words_is_starred 
        ON user_words(user_id) 
        WHERE is_starred = true
    """))

    print("Creating index: idx_user_words_learned_updated...")
    connection.execute(sa.text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_words_learned_updated 
        ON user_words(user_id, updated_at DESC) 
        WHERE is_learned = true
    """))

    print("✅ All performance indexes created successfully")


def downgrade() -> None:
    """Downgrade schema."""
    connection = op.get_bind()

    # Drop indexes (no CONCURRENTLY needed for DROP)
    print("Dropping performance indexes...")

    connection.execute(sa.text("""
        DROP INDEX IF EXISTS idx_user_words_learned_updated
    """))

    connection.execute(sa.text("""
        DROP INDEX IF EXISTS idx_user_words_is_starred
    """))

    connection.execute(sa.text("""
        DROP INDEX IF EXISTS idx_user_words_is_learned
    """))

    connection.execute(sa.text("""
        DROP INDEX IF EXISTS idx_user_words_user_word
    """))

    connection.execute(sa.text("""
        DROP INDEX IF EXISTS idx_user_words_user_id
    """))

    print("🗑️  Performance indexes removed")