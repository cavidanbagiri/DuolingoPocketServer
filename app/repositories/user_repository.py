import os
from typing import Any, Coroutine, Union

from fastapi import HTTPException

from sqlalchemy import insert, select, delete, select, func
from sqlalchemy.exc import NoResultFound, DBAPIError



# services/reset_password_service.py
import secrets
import hashlib
from datetime import datetime, timedelta
from fastapi import HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging




# New added for Google sign in
import httpx
from jose import jwt
from jose.exceptions import JWTError
###########################################

from app.auth.refresh_token_handler import DeleteRefreshTokenRepository
from app.schemas.user_schema import UserLoginSchema

from app.models.user_model import UserModel, TokenModel, UserLanguage, PasswordResetToken, UserWord

from app.utils.hash_password import PasswordHash

from app.auth.token_handler import TokenHandler

# Initialize your Argon2 password hasher
password_hasher = PasswordHash()


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



class GetTotalLearnedRepository:
    def __init__(self, db: AsyncSession, user_id: int):
        self.db = db
        self.user_id = user_id

    async def get_total_learned_words(self):
        # Simple count query for learned words
        stmt = (
            select(func.count(UserWord.id))
            .where(
                UserWord.user_id == self.user_id,
                UserWord.is_learned == True
            )
        )

        result = await self.db.execute(stmt)
        total_learned = result.scalar_one()
        return {"total_learned_words": total_learned}



class ResetPasswordService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def request_password_reset(self, email: str):
        """Handle password reset request"""
        try:

            if await self._is_rate_limited(email):
                logger.warning(f"Rate limit exceeded for email: {email}")
                return self._success_response()  # Still return success for security

            # Find user by email
            user = await self._get_user_by_email(email)
            if not user:
                # For security, don't reveal if email exists
                return self._success_response()

            # Create reset token
            reset_token_obj, plain_token = await self._create_reset_token(user.id)

            # Send reset email
            await self._send_reset_email(user.email, plain_token)

            return self._success_response()

        except Exception as e:
            logger.error(f"Error in request_password_reset: {str(e)}")
            # Still return success for security
            return self._success_response()

    async def confirm_password_reset(self, token: str, new_password: str):
        """Confirm password reset with token"""
        try:
            # Validate token
            reset_token = await self._validate_reset_token(token)
            if not reset_token:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid or expired reset token"
                )

            # Get user
            user = reset_token.user

            # Validate new password
            self._validate_password(new_password)

            # Update user password using Argon2
            user.password = password_hasher.hash_password(new_password)

            # Mark token as used
            reset_token.used = True
            reset_token.used_at = datetime.utcnow()

            await self.db.commit()

            return {
                "message": "Password reset successfully",
                "detail": "You can now login with your new password"
            }

        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error in confirm_password_reset: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reset password. Please try again."
            )

    async def _get_user_by_email(self, email: str) -> UserModel:
        """Find user by email using SQLAlchemy 2.0 style"""
        query = select(UserModel).where(UserModel.email == email)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def _create_reset_token(self, user_id: int):
        """Create a new reset token"""
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        expires_at = datetime.utcnow() + timedelta(hours=1)

        # Create token object
        reset_token = PasswordResetToken(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at
        )

        # Save to database
        self.db.add(reset_token)
        await self.db.commit()

        return reset_token, token

    async def _validate_reset_token(self, token: str) -> PasswordResetToken:
        """Validate reset token and return token object if valid"""
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        query = select(PasswordResetToken).where(
            and_(
                PasswordResetToken.token_hash == token_hash,
                PasswordResetToken.used == False,
                PasswordResetToken.expires_at > datetime.utcnow()
            )
        ).options(selectinload(PasswordResetToken.user))

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    def _validate_password(self, password: str):
        """Validate new password strength"""
        if len(password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters long"
            )
        # Add more password strength checks as needed

    def _success_response(self):
        """Standard success response for security"""
        return {
            "message": "If an account with that email exists, a reset link has been sent",
            "detail": "Check your email for the reset link"
        }

    async def _send_reset_email(self, email: str, token: str):
        """Send password reset email with improved error handling"""
        try:
            # Email configuration
            smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.getenv('SMTP_PORT', 587))
            smtp_username = os.getenv('SMTP_USERNAME')
            smtp_password = os.getenv('SMTP_PASSWORD')
            from_email = os.getenv('FROM_EMAIL', 'noreply@w9999.com')

            # Validate required environment variables
            if not all([smtp_username, smtp_password]):
                logger.error("SMTP credentials not configured properly")
                return  # Don't raise exception - fail silently for security

            # Create reset link
            frontend_url = os.getenv('FRONTEND_URL', 'https://www.w9999.tech')
            reset_link = f"{frontend_url}/reset-password-confirm?token={token}"

            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = "Reset Your Password - W9999"
            msg['From'] = from_email
            msg['To'] = email

            # HTML email content
            html = self._create_email_html(reset_link)
            msg.attach(MIMEText(html, 'html'))

            # Send email with timeout
            logger.info(f"Attempting to send reset email to {email}")

            with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
                server.ehlo()  # Identify ourselves to SMTP server
                server.starttls()  # Secure the connection
                server.ehlo()  # Re-identify ourselves over TLS connection

                # Login and send
                server.login(smtp_username, smtp_password)
                server.send_message(msg)

            logger.info(f"Password reset email successfully sent to {email}")

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP Authentication failed: {str(e)}")
            logger.error("Please check your SMTP credentials and app password")
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error occurred: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error sending email: {str(e)}")

    def _create_email_html(self, reset_link: str) -> str:
        """Create beautiful HTML email template"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Arial', sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; }}
                .container {{ max-width: 600px; margin: 0 auto; background: #ffffff; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; text-align: center; color: white; }}
                .content {{ padding: 40px; background: #f8f9fa; }}
                .button {{ display: inline-block; padding: 14px 35px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; text-decoration: none; border-radius: 25px; font-weight: bold; font-size: 16px; margin: 20px 0; }}
                .footer {{ padding: 20px; text-align: center; color: #666; font-size: 12px; background: #f1f3f4; }}
                .code {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #667eea; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔐 W9999</h1>
                    <p>Password Reset Request</p>
                </div>
                <div class="content">
                    <h2>Hello!</h2>
                    <p>We received a request to reset your password for your W9999 account.</p>
                    <p>Click the button below to reset your password:</p>
                    <p style="text-align: center;">
                        <a href="{reset_link}" class="button">Reset Password</a>
                    </p>
                    <p>Or copy and paste this link in your browser:</p>
                    <div class="code">
                        {reset_link}
                    </div>
                    <p><strong>⚠️ This link will expire in 1 hour.</strong></p>
                    <p>If you didn't request a password reset, please ignore this email.</p>
                    <p>Happy learning!<br>The W9999 Team</p>
                </div>
                <div class="footer">
                    <p>© 2024 W9999. All rights reserved.</p>
                    <p>If you're having trouble, contact us at cavidanbagiri@gmail.com</p>
                </div>
            </div>
        </body>
        </html>
        """


    async def _is_rate_limited(self, email: str) -> bool:
        """Check if too many reset requests for this email (last hour)"""
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)

        query = select(PasswordResetToken).join(UserModel).where(
            and_(
                UserModel.email == email,
                PasswordResetToken.created_at >= one_hour_ago
            )
        )

        result = await self.db.execute(query)
        recent_requests = result.scalars().all()

        # Allow max 3 reset requests per hour
        return len(recent_requests) >= 3