import asyncio
import aiofiles
import uuid
from typing import Annotated, Dict, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, status
from fastapi.responses import Response, JSONResponse

from sqlalchemy.ext.asyncio import AsyncSession


from app.auth.refresh_token_handler import VerifyRefreshTokenMiddleware
from app.auth.token_handler import TokenHandler
from app.database.setup import get_db

from app.repositories.user_repository import (UserRegisterRepository, UserLoginRepository, UserLogoutRepository,
                                              CheckUserAvailable, RefreshTokenRepository, DeleteRefreshTokenRepository,
                                              SetNativeRepository, ChooseLangTargetRepository, GoogleAuthRepository,
                                              GetNativeRepository, EditUserProfileRepository,
                                              ResetPasswordService, GetTotalLearnedRepository,
                                              EditUserProfileImageRepository)

from app.schemas.user_schema import UserLoginSchema, UserTokenSchema, UserRegisterSchema, NativeLangSchema, \
    ChooseLangSchema, EditUserProfileSchema

from app.logging_config import setup_logger
logger = setup_logger(__name__, "user.log")

router = APIRouter()

@router.post('/register', status_code=201)
async def register(response: Response, register_data: UserRegisterSchema,
                   db_session: Annotated[AsyncSession, Depends(get_db)]):
    repository = UserRegisterRepository(db_session)

    try:
        # ADD DETAILED LOGGING
        logger.info(f"🔍 REGISTRATION START - User: {register_data.email}, Selected Native: '{register_data.native}'")

        data = await repository.register(register_data)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"

        response.set_cookie('refresh_token', data.get('refresh_token'),
                            httponly=True,
                            secure=True,
                            samesite="none"
                            )

        logger.info(
            f"🔍 REGISTRATION COMPLETE - User: {register_data.email}, Saved Native: '{data.get('user', {}).get('native')}'")

        # completed_user = await UserLoginRepository.get_user_with_profile(db, int(user_info.get('user').get('sub')))

        return {
            'user': data.get('user'),
            'access_token': data.get('access_token')
        }

    except HTTPException as ex:
        logger.error(f"🔍 REGISTRATION HTTP ERROR - User: {register_data.email}, Error: {ex}")
        raise ex
    except Exception as ex:
        logger.exception(f"🔍 REGISTRATION UNEXPECTED ERROR - User: {register_data.email}, Error: {ex}")
        raise HTTPException(500, 'Internal server error')



@router.post('/login', status_code=201)
async def login(response: Response, login_data: UserLoginSchema, db_session: Annotated[AsyncSession,  Depends(get_db)]):

    repository = UserLoginRepository(db_session)

    try:
        data = await repository.login(login_data)


        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"

        response.set_cookie('refresh_token', data.get('refresh_token'),
                            httponly=True,
                            secure=True,
                            samesite="none"
                            )
        return {
            'user': data.get('user'),
            'access_token': data.get('access_token')
        }

    except HTTPException as ex:
        print(f'error happened {ex}')
        raise ex
    except Exception as ex:  # Catch all other exceptions
        logger.exception("Unexpected error login user: %s", ex)
        raise HTTPException(500, 'Internal server error')



# Google Login
@router.post('/google', status_code=201)
async def google_auth(
        response: Response,
        google_data: dict,  # We'll receive { "code": "authorization_code" }
        db_session: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Google Sign-In endpoint
    Handles both registration and login automatically
    """
    logger.info("Google auth endpoint called")

    try:
        # Validate input
        if not google_data.get('code'):
            logger.error("Google auth called without authorization code")
            raise HTTPException(status_code=400, detail="Authorization code is required")

        # Initialize repository
        repository = GoogleAuthRepository(db_session)

        # Process Google authentication
        logger.info("Processing Google authentication")
        auth_result = await repository.authenticate_with_google(google_data['code'])

        # Set refresh token cookie (same as your existing endpoints)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"

        response.set_cookie(
            key='refresh_token',
            value=auth_result.get('refresh_token'),
            httponly=True,
            secure=True,
            samesite="none"
        )

        logger.info("Google auth completed successfully")
        return {
            'user': auth_result.get('user'),
            'access_token': auth_result.get('access_token')
        }

    except HTTPException as ex:
        logger.error(f"Google auth HTTP error: {ex.detail}")
        raise ex
    except Exception as ex:
        logger.exception(f"Unexpected error in Google auth: {ex}")
        raise HTTPException(500, 'Internal server error')


@router.post("/refresh", status_code=200)
async def refresh_token(response: Response, request: Request, db: AsyncSession = Depends(get_db)):

    middleware = VerifyRefreshTokenMiddleware(db)
    try:
        user_info = await middleware.validate_refresh_token(request, response)
        if not user_info:
            response.delete_cookie(key="refresh_token")
            raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

        completed_user = await UserLoginRepository.get_user_with_profile(db, int(user_info.get('user').get('sub')))

        return {
            "access_token": user_info.get("access_token"),
            "user": completed_user,
        }

    except HTTPException as e:
        logger.error(f"Error refreshing token: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error refreshing token: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while refreshing the token {e}")


@router.post('/logout', status_code=200)
async def logout(
        request: Request,
        response: Response,
        db: AsyncSession = Depends(get_db)
):
    # Get refresh token from cookies
    refresh_token = request.cookies.get("refresh_token")

    if not refresh_token:
        # If no refresh token exists, just return success (already logged out)
        response.delete_cookie(
            key="refresh_token",
            secure=True,
            samesite="none",
            httponly=True,
            path="/"
        )
        return {"message": "Logout successful"}

    user_logout_repository = UserLogoutRepository(db)
    try:
        # Try to verify the refresh token to get user ID
        user_payload = TokenHandler.verify_refresh_token(refresh_token)
        user_id = int(user_payload.get('sub'))

        # Perform server-side logout
        result = await user_logout_repository.logout(user_id)

        # Always clear the cookie regardless of server-side result
        response.delete_cookie(
            key="refresh_token",
            secure=True,
            samesite="none",
            httponly=True,
            path="/"
        )

        return {"message": "Logout successful"}

    except HTTPException as ex:
        # Even if token verification fails, clear the cookie
        response.delete_cookie(
            key="refresh_token",
            secure=True,
            samesite="none",
            httponly=True,
            path="/"
        )
        return {"message": "Logout successful"}

    except Exception as e:
        # Clear cookie on any error
        response.delete_cookie(
            key="refresh_token",
            secure=True,
            samesite="none",
            httponly=True,
            path="/"
        )
        logger.error(f"Error during logout: {str(e)}")
        return {"message": "Logout successful"}



@router.post('/setnative', status_code=201)
async def set_native(
    data: NativeLangSchema,
    db: AsyncSession = Depends(get_db),
    user_info = Depends(TokenHandler.verify_access_token)
):
    try:
        repo = SetNativeRepository(db=db, user_id=int(user_info.get('sub')), native=data.native, )
        result = await repo.set_native()
        return result
    except Exception as ex:
        return {'error': str(ex)}



@router.get('/getnative')
async def get_native_language(
        db: AsyncSession = Depends(get_db),
        user_info=Depends(TokenHandler.verify_access_token)
):
    """
    Check if user has set their native language
    """
    try:
        user_id = int(user_info.get('sub'))

        repo = GetNativeRepository(db)

        result = await repo.get_native(user_id)

        return result

    except Exception as ex:
        logger.error(f"Error checking native language: {str(ex)}")
        raise HTTPException(status_code=500, detail="Internal server error")



@router.post('/choose_lang', status_code=201)
async def choose_target_lang(
    data: ChooseLangSchema,
    db: AsyncSession = Depends(get_db),
    user_info = Depends(TokenHandler.verify_access_token)
):
    try:
        repo = ChooseLangTargetRepository(db=db, target_lang_code=data.target_language_code, user_id=int(user_info.get('sub')))
        result = await repo.choose_lang_target()
        return result
    except Exception as ex:
        return {'error': str(ex)}



@router.get('/total-learned-words')
async def get_total_learned_words(
        db: AsyncSession = Depends(get_db),
        user_info = Depends(TokenHandler.verify_access_token)
):
    try:
        repo = GetTotalLearnedRepository(db=db, user_id=int(user_info.get('sub')))
        result = await repo.get_total_learned_words()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during fetching total learned words size: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="We're having trouble with the fetching total learned words size"
        )


from pydantic import BaseModel, EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordConfirmRequest(BaseModel):
    token: str
    new_password: str

@router.post("/reset-password", status_code=200)
async def reset_password(
    request: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db)
):
    """Request password reset"""
    service = ResetPasswordService(db)
    return await service.request_password_reset(request.email)


@router.post("/reset-password-confirm", status_code=200)
async def reset_password_confirm(
    request: ResetPasswordConfirmRequest,
    db: AsyncSession = Depends(get_db)
):
    """Confirm password reset"""
    service = ResetPasswordService(db)
    return await service.confirm_password_reset(request.token, request.new_password)


@router.post('/edit/update-profile', status_code=200)
async def update_user_profile(
        user_data: EditUserProfileSchema,
        user_info: Dict = Depends(TokenHandler.verify_access_token),
        db: AsyncSession = Depends(get_db)
):
    """
    Update user profile information

    Args:
        user_data: User profile data to update
        user_info: Authenticated user information from token
        db: Database session

    Returns:
        Success message and updated profile data
    """
    try:
        user_id = int(user_info.get('sub'))
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user token"
            )

        repo = EditUserProfileRepository(db=db, user_id=user_id)
        result = await repo.update_user(user_data)

        return result

    except HTTPException as http_exc:
        logger.warning(f"HTTPException during profile update for user {user_info.get('sub')}: {http_exc.detail}")
        raise http_exc
    except ValueError as ve:
        logger.error(f"Validation error during profile update: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during user profile update: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error during user profile update"
        )



@router.post('/edit/update-profile-image', status_code=200)
async def edit_user_profile_image(
        file: UploadFile = File(...),
        user_info: Dict = Depends(TokenHandler.verify_access_token),
        db: AsyncSession = Depends(get_db)
):
    """
    Change the user profile image
    :param file: Uploaded image file
    :param user_info:
    :param db:
    :return:
    """
    try:
        # Validate file type
        allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'image/gif', 'image/webp']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Only JPEG, PNG, GIF, and WebP are allowed."
            )

        # Validate file size (max 5MB)
        max_size = 5 * 1024 * 1024  # 5MB
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning

        if file_size > max_size:
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 5MB."
            )

        repo = EditUserProfileImageRepository(db=db, user_id=int(user_info.get('sub')))
        result = await repo.edit_user_profile_image(file)
        return result
    except HTTPException as http_exc:
        logger.warning(f"HTTPException during profile image update for user {user_info.get('sub')}: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error during user profile image update: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Unexpected error during user profile image update"
        )