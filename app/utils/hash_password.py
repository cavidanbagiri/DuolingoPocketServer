
from argon2 import PasswordHasher


class PasswordHash:

    __slots__ = 'ph'

    def __init__(self):
        self.ph = PasswordHasher()

    def hash_password(self, plain_password: str):
        h_password = self.ph.hash(plain_password)
        return h_password

    def verify(self, h_password: str, plain_password: str):
        try:
            return self.ph.verify(h_password, plain_password)
        except Exception as ex:
            return False
