from pydantic import BaseModel
from datetime import datetime

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

# DATASET
class SalesDataResponse(BaseModel):
    sale_id: int
    tgl_pembayaran: datetime
    status_terakhir: str
    nama_produk: str
    jml_produk_dibeli: int
    harga_jual_idr: int
    total_penjualan_idr: int

    class Config:
        orm_mode = True

class DataSummaryResponse(BaseModel):
    total_transaksi: int
    total_produk: int
    periode_awal: datetime
    periode_akhir: datetime
    total_status_selesai: int
    total_status_dibatalkan: int
    total_status_dibatalkan_pembeli: int
    total_status_dibatalkan_penjual: int
    total_status_dibatalkan_sistem: int
    total_status_sedang_dikirim: int

    class Config:
        orm_mode = True

# TOKEN
class Token(BaseModel):
    access_token: str
    token_type: str

    class Config:
        orm_mode = True