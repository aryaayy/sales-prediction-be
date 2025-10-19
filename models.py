from sqlalchemy import Column, Integer, String, Boolean
from database import Base

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement="auto")
    email = Column(String(255), nullable=False, unique=True, index=True)
    password = Column(String(255), nullable=False)
    nama_lengkap = Column(String(255), nullable=False)
    nama_toko = Column(String(255), nullable=False, default="")
    role = Column(String(50), nullable=False)
