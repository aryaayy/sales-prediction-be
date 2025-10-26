from pydantic import BaseModel
from typing import Optional
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
    invoice: Optional[str] = None
    tanggal_pembayaran: Optional[datetime] = None
    status_terakhir: Optional[str] = None
    nama_produk: Optional[str] = None
    jumlah_produk_dibeli: Optional[int] = None
    harga_jual_idr: Optional[int] = None
    total_penjualan_idr: Optional[int] = None

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

# TOP PRODUCTS
class TopProductsResponse(BaseModel):
    nama_produk: str
    total_unit_terjual: int
    total_transaksi: int
    total_penjualan: int

    class Config:
        orm_mode = True

class TopProductsSummaryResponse(BaseModel):
    total_penjualan_top: int
    total_unit_terjual_top: int

    class Config:
        orm_mode = True

# SALES TREND
class SalesTrendResponse(BaseModel):
    bulan_pembayaran: int
    tahun_pembayaran: int
    total_penjualan: int
    pertumbuhan: float

    class Config:
        orm_mode = True

class TransactionAnalysisResponse(BaseModel):
    avg_penjualan: float
    max_penjualan: int
    min_penjualan: int
    median_penjualan: int
    std_penjualan: float

    class Config:
        orm_mode = True

class TemporalPatternResponse(BaseModel):
    hari_transaksi: int
    jumlah_transaksi_hari: int
    bulan_transaksi: int
    jumlah_transaksi_bulan: int
    rentang_jam_transaksi: str
    jumlah_transaksi_jam: int

    class Config:
        orm_mode = True

# TOKEN
class Token(BaseModel):
    access_token: str
    token_type: str

    class Config:
        orm_mode = True