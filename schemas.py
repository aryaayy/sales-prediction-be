from pydantic import BaseModel

# USER
class UserBase(BaseModel):
    email: str

class UserLogin(UserBase):
    password: str

class UserCreate(UserLogin):
    nama_lengkap: str
    role: str

class UserChangePassword(UserLogin):
    user_id: int
    new_password: str

class UserResponse(UserBase):
    user_id: int
    nama_lengkap: str
    nama_toko: str
    role: str

    class Config:
        orm_mode = True

# TOKEN
class Token(BaseModel):
    access_token: str
    token_type: str

    class Config:
        orm_mode = True