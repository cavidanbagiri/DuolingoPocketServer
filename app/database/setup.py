
import os

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
)

# connection_string = os.getenv('DATABASE_URL')

# temp = os.getenv('DATABASE_URL')
connection_string = 'postgresql+asyncpg://postgres:cavidan1@localhost:5432/linguapocket'

engine = create_async_engine(
    connection_string,
    echo=True,
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=10,
)

# Session factory with expire_on_commit=False for async safety
SessionLocal = async_sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

async def get_db():
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise
        finally:
            await session.close()



