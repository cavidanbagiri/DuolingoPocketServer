#
# from fastapi import APIRouter, HTTPException, status
#
# from app.constants.supported_languages import SUPPORTED_LANGUAGES
# from app.repositories.translate_repository import TranslateRepository
# from app.schemas.translate_schema import TranslateSchema, WordSchema
#
# router = APIRouter()
#
#
# from app.logging_config import setup_logger
# logger = setup_logger(__name__, "translate.log")
#
#
# @router.get("/languages", status_code = status.HTTP_200_OK)
# def get_supported_languages():
#     try:
#         return SUPPORTED_LANGUAGES
#     except Exception as ex:
#         logger.error(f"Get supported languages error {ex}")
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error")
#
#
#
# @router.post("/")
# async def translate(data: TranslateSchema):
#
#     print(data)
#     # Validate input text first
#     if not data.q or not data.q.strip():
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Text to translate cannot be empty"
#         )
#
#     if len(data.q.strip()) < 2:  # Minimum 2 characters
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Text too short to translate"
#         )
#
#     try:
#         repository = TranslateRepository(data)
#         return await repository.translate()
#     except HTTPException as ex:
#         logger.error(f'Translate error HTTP {ex}')
#         raise ex
#     except Exception as ex:
#         logger.error(f'Translate Error {ex}')
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail='Internal Server Error'
#         )
#
# #
# # @router.post('/save', status_code=201)
# # async def save_word(data: WordSchema,
# #                     db:Annotated[AsyncSession, Depends(get_db)],
# #                     user_info = Depends(TokenHandler.verify_access_token)
# #                     ):
# #     try:
# #         repository = SaveWordRepository(data, user_info.get('sub'), db)
# #         return_data = await repository.save_word()
# #         return {"msg":"created","data":return_data}
# #     except HTTPException as ex:
# #         raise ex
# #     except Exception as ex:
# #         logger.error(f'Translate Error {ex}')
# #         raise HTTPException(
# #             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
# #             detail='Internal Server Error'
# #         )
# #
