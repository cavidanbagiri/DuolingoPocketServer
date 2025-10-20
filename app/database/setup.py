#
# import os
#
# from sqlalchemy.ext.asyncio import (
#     create_async_engine,
#     async_sessionmaker,
# )
#
# connection_string = 'postgresql+asyncpg://postgres:cavidan1@localhost:5432/linguapocket'
# # connection_string = 'w9999-user:<User_password>@rc1a-ja4egdn663cqmc3m.mdb.yandexcloud.net:6432/W9999-Database'
#
# engine = create_async_engine(
#     connection_string,
#     echo=True,
#     pool_pre_ping=True,
#     pool_size=20,
#     max_overflow=10,
# )
#
# # Session factory with expire_on_commit=False for async safety
# SessionLocal = async_sessionmaker(
#     bind=engine,
#     autocommit=False,
#     autoflush=False,
#     expire_on_commit=False,
# )
#
# async def get_db():
#     async with SessionLocal() as session:
#         try:
#             yield session
#             await session.commit()
#         except Exception as e:
#             await session.rollback()
#             raise
#         finally:
#             await session.close()
#
#





# production
import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database credentials
# DB_USER = os.getenv('DB_USER')
# DB_PASSWORD = os.getenv('DB_PASSWORD')
# DB_HOST = os.getenv('DB_HOST')
# DB_PORT = os.getenv('DB_PORT')
# DB_NAME = os.getenv('DB_NAME')

DB_USER = os.getenv('RENDER_DB_USER')
DB_PASSWORD = os.getenv('RENDER_DB_PASSWORD')
DB_HOST = os.getenv('RENDER_DB_HOST')
DB_PORT = os.getenv('RENDER_DB_PORT')
DB_NAME = os.getenv('RENDER_DB_NAME')


connection_string = f'postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Alternative connection approach with explicit parameters
engine = create_async_engine(
    connection_string,
    echo=True,
    pool_pre_ping=True,
    pool_size=5,  # Reduced for testing
    max_overflow=5,
    # connect_args={
    #     "ssl": "require",
    #     "server_settings": {
    #         "application_name": "linguapocket_app"
    #     }
    # }
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

