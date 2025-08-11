import os
from datetime import datetime, timezone, timedelta

from fastapi.requests import Request

import jwt
from jwt.exceptions import InvalidTokenError

from fastapi import HTTPException

from app.logging_config import setup_logger
logger = setup_logger(__name__, "token.log")

class TokenHandler:

    @staticmethod
    def generate_access_token(user_data: dict) -> str:
        try:
            encode = user_data.copy()
            encode.update(({"exp": datetime.now(timezone.utc) + timedelta(days=2)}))
            secret_key = os.getenv('JWT_SECRET_KEY')
            algorithm = os.getenv('JWT_ALGORITHM')
            access_token = jwt.encode(encode, secret_key, algorithm)
            return access_token
        except Exception as ex:
            logger.error(f"Failed to created new access token {ex}")
            raise HTTPException(status_code=500, detail=f"Failed to created new access token {ex}")

    @staticmethod
    def generate_refresh_token(user_data) -> str:
        try:
            encode = user_data.copy()
            encode.update(({"exp": datetime.now(timezone.utc) + timedelta(days=30)}))
            secret_key = os.getenv('JWT_REFRESH_SECRET_KEY')
            algorithm = os.getenv('JWT_ALGORITHM')
            refresh_token = jwt.encode(encode, secret_key, algorithm)
            return refresh_token

        except Exception as ex:
            logger.error(f"Failed to created new access token {ex}")
            raise HTTPException(status_code=500, detail=f"Failed to created new access token {ex}")

    @staticmethod
    def verify_access_token(req: Request) -> dict:
        print(f'req is {req.headers}')
        if req.headers.get('Authorization'):
            access_token = req.headers.get('Authorization').split(' ')[1]
            if access_token:
                try:
                    secret_key = os.getenv('JWT_SECRET_KEY')
                    algorithm = os.getenv('JWT_ALGORITHM')
                    payload = jwt.decode(access_token, secret_key, algorithm)
                    return payload
                except InvalidTokenError as ex:
                    raise HTTPException(status_code=401, detail=f'Authorization Error {ex}')
            else:
                raise HTTPException(status_code=401, detail='Authorization Error')
        else:
            raise HTTPException(status_code=401, detail='Authorization Error')
