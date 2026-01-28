"""create_index_for_wordlanguage_frequencyrankss

Revision ID: 8a5e737dfd89
Revises: 3e593903ea75
Create Date: 2026-01-27 23:52:46.609906

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8a5e737dfd89'
down_revision: Union[str, None] = '3e593903ea75'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    connection = op.get_bind()
    connection.execute(sa.text("COMMIT"))

    # ONLY 2 MOST CRITICAL INDEXES for your query:

    # 1. Covering index for words table (biggest impact)
    connection.execute(sa.text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_words_query_opt 
        ON words(language_code, frequency_rank) 
        INCLUDE (text, level, id)
    """))

    connection.execute(sa.text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_translations_query_opt 
        ON translations(source_word_id, target_language_code) 
        INCLUDE (translated_text)
        """))

    print("✅ Critical query indexes created")


def downgrade() -> None:
    """Downgrade schema."""
    connection = op.get_bind()

    connection.execute(sa.text("""
        DROP INDEX IF EXISTS idx_translations_query_opt
    """))

    connection.execute(sa.text("""
        DROP INDEX IF EXISTS idx_words_query_opt
    """))