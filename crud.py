from sqlalchemy.orm import Session
import models, schemas
from dotenv import load_dotenv
import bcrypt, os

# USERS
def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def hash_password(password):
    load_dotenv()
    SALT = os.getenv("SALT").encode('utf-8')
    byte_password = password.encode('utf-8')
    hashed = bcrypt.hashpw(byte_password, SALT).decode('utf-8')
    return hashed

def insert_user(db: Session, user: schemas.UserCreate):
    hashed_pwd = hash_password(user.password)
    db_user = models.User(
        email=user.email,
        password=hashed_pwd,
        nama_lengkap=user.nama_lengkap,
        role=user.role
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

def update_user(db: Session, user: schemas.UserResponse):
    db_user = db.query(models.User).filter(models.User.user_id == user.user_id).first()

    if not db_user:
        return None
    
    db_user.email = user.email
    db_user.nama_lengkap = user.nama_lengkap
    db_user.nama_toko = user.nama_toko

    db.commit()
    db.refresh(db_user)

    return db_user