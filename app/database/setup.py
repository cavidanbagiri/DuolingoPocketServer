

# production
import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# DB_USER = os.getenv('RENDER_DB_USER')
# DB_PASSWORD = os.getenv('RENDER_DB_PASSWORD')
# DB_HOST = os.getenv('RENDER_DB_HOST')
# DB_PORT = os.getenv('RENDER_DB_PORT')
# DB_NAME = os.getenv('RENDER_DB_NAME')


DB_USER = os.getenv('YANDEX_DB_USER')
DB_PASSWORD = os.getenv('YANDEX_DB_PASSWORD')
DB_HOST = os.getenv('YANDEX_DB_HOST')
DB_PORT = os.getenv('YANDEX_DB_PORT')
DB_NAME = os.getenv('YANDEX_DB_NAME')


connection_string = f'postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Alternative connection approach with explicit parameters
# engine = create_async_engine(
#     connection_string,
#     echo=True,
#     pool_pre_ping=True,
#     pool_size=5,  # Reduced for testing
#     max_overflow=5,
# )




engine = create_async_engine(
    connection_string,
    echo=False,  # Set to False for production/load testing
    pool_pre_ping=True,
    pool_size=20,       # Increased from 5
    max_overflow=30,    # Increased from 5
    pool_recycle=3600,  # Recycle connections every hour
    pool_timeout=30,    # Wait 30 seconds for connection
    # max_connections=50, # Total limit
)



# Session factory
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
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()

