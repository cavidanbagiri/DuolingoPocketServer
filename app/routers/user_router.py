from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from fastapi.responses import Response, JSONResponse

from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.refresh_token_handler import VerifyRefreshTokenMiddleware
from app.auth.token_handler import TokenHandler
from app.database.setup import get_db

from app.repositories.user_repository import UserRegisterRepository, UserLoginRepository, UserLogoutRepository
from app.schemas.user_schema import UserLoginSchema, UserTokenSchema, UserRegisterSchema

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
        raise ex
    except Exception as ex:  # Catch all other exceptions
        logger.exception("Unexpected error login user: %s", ex)
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
        print(f"An error occurred while refreshing the token {e}")
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