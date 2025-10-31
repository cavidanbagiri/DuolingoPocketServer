import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from fastapi.responses import Response, JSONResponse

from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.refresh_token_handler import VerifyRefreshTokenMiddleware
from app.auth.token_handler import TokenHandler
from app.database.setup import get_db

from app.repositories.user_repository import (UserRegisterRepository, UserLoginRepository, UserLogoutRepository,
                                              CheckUserAvailable, RefreshTokenRepository, DeleteRefreshTokenRepository,
                                              SetNativeRepository, ChooseLangTargetRepository, GoogleAuthRepository)

from app.schemas.user_schema import UserLoginSchema, UserTokenSchema, UserRegisterSchema, NativeLangSchema, \
    ChooseLangSchema

from app.logging_config import setup_logger
logger = setup_logger(__name__, "user.log")

router = APIRouter()


@router.post('/register', status_code=201)
async def register(response: Response, register_data: UserRegisterSchema,
                   db_session: Annotated[AsyncSession, Depends(get_db)]):
    repository = UserRegisterRepository(db_session)

    try:
        data = await repository.register(register_data)

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
        raise ex
    except Exception as ex:  # Catch all other exceptions
        logger.exception("Unexpected error login user: %s", ex)
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

        return {
            "access_token": user_info.get("access_token"),
            "user": user_info.get('user'),
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
    user_payload: Annotated[UserTokenSchema, Depends(TokenHandler.verify_access_token)],
    db: AsyncSession = Depends(get_db)
):
    if not user_payload:
        return JSONResponse(status_code=401, content={"message": "Please login before logging out"})
    user_logout_repository = UserLogoutRepository(db)
    try:
        result = await user_logout_repository.logout(int(user_payload.get('sub')))

        if result:
            response.delete_cookie(key="refresh_token")
            return{
                "message": "Logout successful"
            }

        return JSONResponse(status_code=500, content={"message": "Error logging out user"})
    except HTTPException as ex:
        return JSONResponse(status_code=200, content={"message": f"An error occurred during logout {ex}"})
    except Exception as e:
        logger.error(f"Error during logout: {str(e)}")
        return JSONResponse(status_code=500, content={"message": f"An error occurred during logout {e}"})




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

