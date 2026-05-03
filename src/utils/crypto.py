from cryptography.fernet import Fernet
from core.config import config

def get_fernet() -> Fernet:
    key = config.EMAIL_ENCRYPTION_KEY
    if not key:
        raise ValueError("EMAIL_ENCRYPTION_KEY is not set in the environment.")
    return Fernet(key.encode("utf-8"))

def encrypt_password(plain_text: str) -> str:
    """Encrypts a plaintext password and returns the encrypted string."""
    if not plain_text:
        return ""
    f = get_fernet()
    encrypted = f.encrypt(plain_text.encode("utf-8"))
    return encrypted.decode("utf-8")

def decrypt_password(encrypted_text: str) -> str:
    """Decrypts an encrypted password string back to plaintext."""
    if not encrypted_text:
        return ""
    f = get_fernet()
    decrypted = f.decrypt(encrypted_text.encode("utf-8"))
    return decrypted.decode("utf-8")
