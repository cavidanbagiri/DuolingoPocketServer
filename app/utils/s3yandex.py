
import os

# Import Yandex Storage and create connection
from boto3.session import Session


s3 = Session(
    aws_access_key_id=os.getenv('YANDEX_STORAGE_KEY_ID'),
    aws_secret_access_key=os.getenv('YANDEX_STORAGE_SECRET_KEY'),
    region_name='ru-central1'
).client('s3', endpoint_url='https://storage.yandexcloud.net')
