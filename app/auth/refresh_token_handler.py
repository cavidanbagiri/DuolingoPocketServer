import os
from datetime import datetime, timezone, timedelta

import jwt

from fastapi import Request, HTTPException

from sqlalchemy import update, delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import Response

from app.auth.token_handler import TokenHandler

from app.models.user_model import TokenModel

from app.logging_config import setup_logger
user_logger = setup_logger(__name__, "user.log")


class VerifyRefreshTokenMiddleware:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def validate_refresh_token(self, request: Request, response: Response):

        refresh_token = request.cookies.get('refresh_token')

        if not refresh_token:
            user_logger.warning("Please login before executing")
            raise HTTPException(status_code=400, detail="Please login before executing")

        try:

            # Check if the new refresh token already exists in the database
            user_id = await SearchRefreshTokenRepository(self.db).search_refresh_token(refresh_token)

            if user_id is None:
                raise HTTPException(status_code=401, detail="Invalid or revoked refresh token")


            # Decode the refresh token
            payload = jwt.decode(
                refresh_token,
                os.getenv('JWT_REFRESH_SECRET_KEY'),
                algorithms=[os.getenv('JWT_ALGORITHM')]
            )

            # Access token always updated
            access_token = TokenHandler.generate_access_token(payload)

            # Refresh token will check expiry date, less than 7 days, will change
            exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
            now = datetime.now(timezone.utc)
            if exp - now < timedelta(days=7):
                new_refresh_token = TokenHandler.generate_refresh_token(payload)
                await UpdateRefreshTokenRepository(self.db).update_refresh_token(int(payload['sub']), new_refresh_token)

                response.set_cookie(
                    key="refresh_token",
                    value=new_refresh_token,
                    httponly=True,
                    secure=True,  # Ensure this is True in production
                    samesite="none",
                )

            else:
                new_refresh_token = refresh_token

            return {
                'access_token': access_token,
                'refresh_token': new_refresh_token,
                'user': payload
            }


        except jwt.ExpiredSignatureError:
            user_logger.warning("Expired refresh token")
            raise HTTPException(status_code=401, detail="Expired refresh token")

        except jwt.InvalidTokenError as e:
            user_logger.warning(f"Invalid refresh token: {str(e)}")
            raise HTTPException(status_code=401, detail="Invalid refresh token")

        except Exception as e:
            user_logger.error(f"Unexpected error validating refresh token: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An error occurred while validating the refresh token, {e}")


class UpdateRefreshTokenRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def update_refresh_token(self, user_id: int, refresh_token: str):
        """
        Updates a refresh token for a user.

        :param user_id: ID of the user.
        :param refresh_token: Refresh token to update.
        """
        try:
            await self.db.execute(
                update(TokenModel).where(TokenModel.user_id == user_id).values(tokens=refresh_token)
            )
            await self.db.commit()
        except Exception as e:
            user_logger.error(f"Error updating refresh token: {str(e)}")
            raise HTTPException (status_code=404, detail = f"Error updating refresh token: {str(e)}")




class DeleteRefreshTokenRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def delete_refresh_token(self, user_id: int) -> None:

        try:

            result = await self.db.execute(
                select(TokenModel).where(TokenModel.user_id == user_id)
            )

            token = result.scalars().first()

            if not token:
                user_logger.warning(f"No refresh token found for user {user_id} (already logged out?)")
                return

            # Delete Token
            await self.db.execute(
                delete(TokenModel).where(TokenModel.user_id == user_id)
            )
            await self.db.commit()
            user_logger.info(f"Refresh token deleted for user {user_id}.")

        except Exception as e:
            user_logger.error(f"Error deleting refresh token for user {user_id}: {str(e)}")
            raise


class SearchRefreshTokenRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def search_refresh_token(self, refresh_token: str):

        try:
            token = await self.db.execute(
                select(TokenModel).where(TokenModel.tokens == refresh_token)
            )

            token = token.scalar()
            if token:
                return token.user_id
            else:
                return None

        except Exception as e:
            user_logger.error(f"Error searching refresh token: {str(e)}")
            raise