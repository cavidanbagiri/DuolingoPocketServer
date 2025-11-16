import os
from datetime import datetime
from typing import Any, Coroutine, Union

from fastapi import HTTPException

from sqlalchemy import insert, select, delete
from sqlalchemy.exc import NoResultFound, DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

# New added for Google sign in
import httpx
from jose import jwt
from jose.exceptions import JWTError
###########################################

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


# class UserRegisterRepository:
#
#     def __init__(self, db: AsyncSession):
#         self.db = db
#         self.refresh_token_repo = RefreshTokenRepository(self.db)
#         self.h_password = PasswordHash()
#
#     async def register(self, register_data):
#
#         try:
#             # 1 - Check user email is available or not
#             data = await self.db.execute(select(UserModel).where(UserModel.email == register_data.email))
#             user = data.scalar()
#             if user:
#                 raise HTTPException(status_code=409, detail="This email already available")
#
#             # 1 - Check user username is available or not
#             if register_data.username:
#                 data = await self.db.execute(select(UserModel).where(UserModel.username == register_data.username))
#                 user = data.scalar()
#                 if user:
#                     raise HTTPException(status_code=409, detail="This username already available")
#
#             register_data.password = self.h_password.hash_password(register_data.password)
#             return_data = await self.save_user(register_data)
#             return return_data
#
#         except HTTPException as ex:
#             raise
#
#         except Exception as ex:
#             raise HTTPException(status_code=404, detail=f"Registration error {ex}")
#
#     async def save_user(self, register_data):
#         user = UserModel(
#             email=register_data.email,
#             password=register_data.password,
#             username=register_data.username,
#             native = register_data.native,
#         )
#         self.db.add(user)
#         await self.db.commit()
#         await self.db.refresh(user)
#
#         token_data = {
#             'sub': str(user.id),
#             'username': user.username,
#         }
#
#         access_token = TokenHandler.generate_access_token(token_data)
#         refresh_token = TokenHandler.generate_refresh_token(token_data)
#
#         await self.refresh_token_repo.manage_refresh_token(user.id, refresh_token)
#
#         return {
#             'user': {
#                 'sub': str(user.id),
#                 'email': user.email,
#                 'username': user.username,
#                 'native': user.native,
#             },
#             'access_token': access_token,
#             'refresh_token': refresh_token
#         }
#

class UserRegisterRepository:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.refresh_token_repo = RefreshTokenRepository(self.db)
        self.h_password = PasswordHash()

    async def register(self, register_data):
        try:
            logger.info(f"🔍 REPOSITORY - Checking email: {register_data.email}, Native: '{register_data.native}'")

            # 1 - Check user email is available or not
            data = await self.db.execute(select(UserModel).where(UserModel.email == register_data.email.lower()))
            user = data.scalar()
            if user:
                raise HTTPException(status_code=409, detail="This email already available")

            # 2 - Check user username is available or not
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
        # LOG EXACTLY WHAT WE'RE SAVING
        logger.info(f"🔍 SAVE_USER - Creating user with native: '{register_data.native}'")

        user = UserModel(
            email=register_data.email.lower(),
            password=register_data.password,
            username=register_data.username,
            native=register_data.native,
        )

        logger.info(f"🔍 SAVE_USER - UserModel object created with native: '{user.native}'")

        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)

        # VERIFY WHAT WAS ACTUALLY SAVED IN DATABASE
        logger.info(f"🔍 SAVE_USER - User saved to DB - ID: {user.id}, Native: '{user.native}'")

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
                'native': user.native,  # This should match what was saved
            },
            'access_token': access_token,
            'refresh_token': refresh_token
        }





class CheckUserAvailable:

    def __init__(self, db: AsyncSession):
        self.db = db
        self.h_password = PasswordHash()

    async def check_user_exists(self, login_data: UserLoginSchema) -> UserModel:
        data = await self.db.execute(select(UserModel).where(UserModel.email==login_data.email.lower()))
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


            # Fetch existing target languages --->>>> New Added
            stmt = select(UserLanguage.target_language_code).where(UserLanguage.user_id == user.id)
            result = await self.db.execute(stmt)
            target_langs = [row[0] for row in result.all()]  # extract codes as list

            token_data = {
                'sub': str(user.id),
                'username': user.username,
                'target_langs': target_langs,
                'native': user.native,
            }

            access_token = TokenHandler.generate_access_token(token_data)
            refresh_token = TokenHandler.generate_refresh_token(token_data)

            await self.refresh_token_repo.manage_refresh_token(user.id, refresh_token)

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


# This is new added
class GoogleAuthRepository:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.refresh_token_repo = RefreshTokenRepository(db)

        # Google OAuth endpoints
        self.token_endpoint = "https://oauth2.googleapis.com/token"
        self.userinfo_endpoint = "https://www.googleapis.com/oauth2/v3/userinfo"

        # Get credentials from environment
        self.client_id = os.getenv('GOOGLE_OAUTH2_CLIENT_ID')
        self.client_secret = os.getenv('GOOGLE_OAUTH2_CLIENT_SECRET')

    async def authenticate_with_google(self, authorization_code: str):
        """
        Main method that handles the entire Google authentication flow
        """
        logger.info("Starting Google authentication process")

        try:
            # Step 1: Exchange authorization code for access token
            logger.info("Exchanging authorization code for access token")
            google_tokens = await self._exchange_code_for_tokens(authorization_code)

            # Step 2: Get user info from Google
            logger.info("Fetching user info from Google")
            google_user_info = await self._get_google_user_info(google_tokens['access_token'])

            # Step 3: Find or create user in our database
            logger.info(f"Looking up user with email: {google_user_info['email']}")
            user = await self._find_or_create_user(google_user_info)

            # Step 4: Generate our JWT tokens
            logger.info(f"Generating JWT tokens for user: {user.email}")
            tokens = await self._generate_our_tokens(user)

            logger.info("Google authentication completed successfully")
            return tokens

        except HTTPException:
            # Re-raise HTTP exceptions (like 401, 404, etc.)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Google authentication: {str(e)}")
            raise HTTPException(status_code=500, detail="Google authentication failed")

    async def _exchange_code_for_tokens(self, authorization_code: str) -> dict:
        """
        Step 1: Exchange the authorization code for Google access tokens
        This is where we prove to Google that we have a valid code
        """
        logger.info("Exchanging authorization code with Google...")

        try:
            async with httpx.AsyncClient() as client:
                # Prepare the request to Google's token endpoint
                data = {
                    'code': authorization_code,
                    'client_id': self.client_id,
                    'client_secret': self.client_secret,
                    'redirect_uri': 'postmessage',  # For web applications
                    'grant_type': 'authorization_code'
                }

                logger.debug(f"Sending token exchange request to Google")
                response = await client.post(self.token_endpoint, data=data)

                if response.status_code != 200:
                    logger.error(f"Google token exchange failed: {response.text}")
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid authorization code"
                    )

                tokens = response.json()
                logger.info("Successfully exchanged code for Google tokens")
                return tokens

        except httpx.RequestError as e:
            logger.error(f"Network error during token exchange: {str(e)}")
            raise HTTPException(status_code=503, detail="Cannot connect to Google services")

    async def _get_google_user_info(self, access_token: str) -> dict:
        """
        Step 2: Use the access token to get user information from Google
        This verifies the user's identity and gets their profile data
        """
        logger.info("Fetching user info from Google...")

        try:
            async with httpx.AsyncClient() as client:
                headers = {'Authorization': f'Bearer {access_token}'}
                response = await client.get(self.userinfo_endpoint, headers=headers)

                if response.status_code != 200:
                    logger.error(f"Failed to fetch user info: {response.text}")
                    raise HTTPException(
                        status_code=400,
                        detail="Failed to get user information from Google"
                    )

                user_info = response.json()
                logger.info(f"Retrieved user info: {user_info['email']}")

                # Validate essential fields
                if not user_info.get('email'):
                    logger.error("Google user info missing email")
                    raise HTTPException(
                        status_code=400,
                        detail="Google account email is required"
                    )

                return user_info

        except httpx.RequestError as e:
            logger.error(f"Network error fetching user info: {str(e)}")
            raise HTTPException(status_code=503, detail="Cannot connect to Google services")

    async def _find_or_create_user(self, google_user_info: dict) -> UserModel:
        """
        Step 3: Find existing user or create new one
        This is where we integrate with your existing user system
        """
        email = google_user_info['email']

        # Check if user already exists
        logger.info(f"Checking if user exists with email: {email}")
        result = await self.db.execute(
            select(UserModel).where(UserModel.email == email)
        )
        existing_user = result.scalar()

        if existing_user:
            logger.info(f"User found in database: {email}")
            return existing_user

        # Create new user (Auto-register)
        logger.info(f"Creating new user for: {email}")
        new_user = UserModel(
            email=email,
            username=self._generate_username(google_user_info),
            # Google users don't have passwords in our system
            password="",  # Or you can generate a random password
            native=None,  # User can set this later
        )

        self.db.add(new_user)
        await self.db.commit()
        await self.db.refresh(new_user)

        logger.info(f"Successfully created new user: {new_user.id}")
        return new_user

    def _generate_username(self, google_user_info: dict) -> str:
        """
        Generate a username from Google profile data
        Uses email prefix or name fields
        """
        # Try to use the name from Google profile
        if google_user_info.get('name'):
            base_username = google_user_info['name'].lower().replace(' ', '_')
        else:
            # Use email prefix (before @)
            base_username = google_user_info['email'].split('@')[0]

        # You might want to add uniqueness check here
        return base_username

    async def _generate_our_tokens(self, user: UserModel) -> dict:
        """
        Step 4: Generate our JWT tokens (same format as your existing auth)
        This integrates with your current token system
        """
        logger.info(f"Generating JWT tokens for user ID: {user.id}")

        # Create token payload (same as your existing system)
        token_data = {
            'sub': str(user.id),
            'username': user.username,
        }

        # Generate tokens using your existing TokenHandler
        access_token = TokenHandler.generate_access_token(token_data)
        refresh_token = TokenHandler.generate_refresh_token(token_data)

        # Save refresh token (same as your existing system)
        await self.refresh_token_repo.manage_refresh_token(user.id, refresh_token)

        # Get user's target languages (same as your login)
        stmt = select(UserLanguage.target_language_code).where(UserLanguage.user_id == user.id)
        result = await self.db.execute(stmt)
        target_langs = [row[0] for row in result.all()]

        # Return same format as your login/register endpoints
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



class GetNativeRepository:
    def __init__(self, db: AsyncSession):
        self.db = db


    async def get_native(self, user_id: int):
        try:

            # user_id = int(user_id)

            result = await self.db.execute(
                select(UserModel.native).where(UserModel.id == user_id)
            )
            native_language = result.scalar_one_or_none()

            data =  {
                'has_native': native_language is not None and native_language != '',
                'native_language': native_language
            }


            return data


        except Exception as ex:
            logger.error(f"Error checking native language: {str(ex)}")
            raise HTTPException(status_code=500, detail="Internal server error")



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
