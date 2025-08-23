from datetime import datetime
from typing import Any, Coroutine, Union

from fastapi import HTTPException

from sqlalchemy import insert, select, delete
from sqlalchemy.exc import NoResultFound, DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.refresh_token_handler import DeleteRefreshTokenRepository
from app.schemas.user_schema import UserLoginSchema

from app.models.user_model import UserModel, TokenModel, UserLanguage

from app.utils.hash_password import PasswordHash

from app.auth.token_handler import TokenHandler


from app.logging_config import setup_logger
logger = setup_logger(__name__, "user.log")


class RefreshTokenRepository:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def manage_refresh_token(self, user_id:int, refresh_token) -> None:

        try:
            token = await self.find_refresh_token(user_id)
            if token:
                await self.delete_refresh_token(user_id)
            await self.save_refresh_token(user_id, refresh_token)
        except Exception as ex:
            logger.error(f'For {user_id}, manage refresh token error {ex}')
            raise HTTPException(status_code=404, detail=f'Manage refresh token error ')

    async def find_refresh_token(self, user_id: int) -> Union[TokenModel, None]:
        try:
            token = await self.db.execute(select(TokenModel).where(TokenModel.user_id == user_id))
            data = token.scalar()
            if data:
                return data
            else:
                return None
        except Exception as ex:
            logger.error(f'For {user_id}, refresh token not found {ex}')
            raise HTTPException(status_code=404, detail=f"Refresh token not found")

    async def delete_refresh_token(self, user_id: int) -> None:
        try:
            await self.db.execute(delete(TokenModel).where(TokenModel.user_id == user_id))
        except Exception as ex:
            logger.error(f'For {user_id},  not found in token model {ex}')
            raise HTTPException(status_code=404, detail=f'User id not found ')

    async def save_refresh_token(self,user_id: int, refresh_token: str) :
        try:
            if refresh_token:
                await self.db.execute(insert(TokenModel).values(
                    user_id = user_id,
                    tokens = refresh_token
                ))
                await self.db.commit()
            else:
                logger.error(f'For {user_id}, Refresh token can\'t find ')
                raise ValueError("Refresh token can't find")
        except Exception as ex:
            logger.error(f'For {user_id}, Refresh token can\'t save {str(ex)}')
            raise HTTPException(status_code=404, detail=f'Refresh Token can\'t save {str(ex)}')


class UserRegisterRepository:

    def __init__(self, db: AsyncSession):
        self.db = db
        self.refresh_token_repo = RefreshTokenRepository(self.db)
        self.h_password = PasswordHash()

    async def register(self, register_data):

        try:
            # 1 - Check user email is available or not
            data = await self.db.execute(select(UserModel).where(UserModel.email == register_data.email))
            user = data.scalar()
            if user:
                raise HTTPException(status_code=409, detail="This email already available")

            # 1 - Check user username is available or not
            if register_data.username:
                data = await self.db.execute(select(UserModel).where(UserModel.username == register_data.username))
                user = data.scalar()
                if user:
                    raise HTTPException(status_code=409, detail="This username already available")

            register_data.password = self.h_password.hash_password(register_data.password)
            return_data = await self.save_user(register_data)
            return return_data

        except HTTPException as ex:
            raise

        except Exception as ex:
            raise HTTPException(status_code=404, detail=f"Registration error {ex}")

    async def save_user(self, register_data):
        user = UserModel(
            email=register_data.email,
            password=register_data.password,
            username=register_data.username
        )
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)

        print(f'the user is {user}')

        token_data = {
            'sub': str(user.id),
            'username': user.username,
        }

        access_token = TokenHandler.generate_access_token(token_data)
        refresh_token = TokenHandler.generate_refresh_token(token_data)

        await self.refresh_token_repo.manage_refresh_token(user.id, refresh_token)

        return {
            'user': {
                'sub': str(user.id),
                'email': user.email,
                'username': user.username,
                'native': user.native,
            },
            'access_token': access_token,
            'refresh_token': refresh_token
        }


class CheckUserAvailable:

    def __init__(self, db: AsyncSession):
        self.db = db
        self.h_password = PasswordHash()

    async def check_user_exists(self, login_data: UserLoginSchema) -> UserModel:
        data = await self.db.execute(select(UserModel).where(UserModel.email==login_data.email))
        user = data.scalar()
        if user:
            logger.info(f'{login_data.email} find in database')
            pass_verify = self.h_password.verify(user.password, login_data.password)
            if pass_verify:
                return user
            else:
                logger.error(f'{login_data.email} password is wrong')
                raise HTTPException(status_code=404, detail="Password is wrong")
        else:
            logger.error(f'{login_data.email} email is wrong')
            raise HTTPException(status_code=404, detail="User not found")


class UserLoginRepository:

    def __init__(self, db: AsyncSession):
        self.db = db
        self.check_user_available = CheckUserAvailable(self.db)
        self.refresh_token_repo = RefreshTokenRepository(self.db)


    async def login(self, login_data: UserLoginSchema)-> dict:
        try:
            logger.info(f'{login_data.email} try to login')
            user = await self.check_user_available.check_user_exists(login_data)
            print(f'second user us {user}')
            token_data = {
                'sub': str(user.id),
                'username': user.username,
            }

            access_token = TokenHandler.generate_access_token(token_data)
            refresh_token = TokenHandler.generate_refresh_token(token_data)

            await self.refresh_token_repo.manage_refresh_token(user.id, refresh_token)

            # Fetch existing target languages --->>>> New Added
            stmt = select(UserLanguage.target_language_code).where(UserLanguage.user_id == user.id)
            result = await self.db.execute(stmt)
            target_langs = [row[0] for row in result.all()]  # extract codes as list

            return self.return_data(user, access_token, refresh_token, target_langs)

        except HTTPException as ex:
            raise

    @staticmethod
    def return_data(user: UserModel, access_token: str, refresh_token: str, target_langs: list[str]):

        return {
            'user': {
                'sub': str(user.id),
                'email': user.email,
                'username': user.username,
                'native': user.native,
                'learning_targets': target_langs
            },
            'access_token': access_token,
            'refresh_token': refresh_token
        }


# Checked - Wrong work
class UserLogoutRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def logout(self, user_id: int):
        """
        Logs out a user by deleting their refresh token.

        :param user_id: ID of the user.
        :return: True if logout is successful, False otherwise.
        """
        try:
            await DeleteRefreshTokenRepository(self.db).delete_refresh_token(user_id)
            logger.info(f"User {user_id} logged out successfully.")
            return {"detail": "Logged out"}
        except NoResultFound:
            logger.warning(f"User {user_id} tried to log out but no token found.")
            return {"detail": "Already logged out"}
        except DBAPIError as e:
            logger.error(f"Database error during logout: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error during logout")
        except Exception as e:
            logger.exception(f"Unexpected error during logout {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="An unexpected error occurred")


class SetNativeRepository:

    def __init__(self, db: AsyncSession, user_id: int, native: str):
        self.db = db
        self.user_id = user_id
        self.native = native

    async def set_native(self):
        try:
            result = await self.db.execute(select(UserModel).where(UserModel.id == self.user_id))
            user = result.scalar_one_or_none()

            if not user:
                return {'error': 'User not found'}

            user.native = self.native  # Set the new native language
            await self.db.commit()
            await self.db.refresh(user)

            return {
                'message': f'Native language set to {self.native}',
                'native': self.native
                }

        except Exception as e:
            await self.db.rollback()
            return {'error': str(e)}



class ChooseLangTargetRepository:

    def __init__(self, db: AsyncSession, target_lang_code: str, user_id: int):
        self.db = db
        self.target_lang_code = target_lang_code
        self.user_id = user_id

    async def choose_lang_target(self):
        lang_map = {
            'en': 'English',
            'ru': 'Russian',
            'tr': 'Turkish'
        }

        # Check if the same language already exists for this user
        stmt = select(UserLanguage).where(
            UserLanguage.user_id == self.user_id,
            UserLanguage.target_language_code == self.target_lang_code
        )
        result = await self.db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            return {
                "msg": "Language already added.",
                "target_language_code": self.target_lang_code
            }

        # Add new target language
        new_pref = UserLanguage(
            user_id=self.user_id,
            target_language_code=self.target_lang_code,
            updated_at=datetime.utcnow()
        )
        self.db.add(new_pref)
        await self.db.commit()

        return {
            "msg": "New language added.",
            "target_language_code": self.target_lang_code
        }
