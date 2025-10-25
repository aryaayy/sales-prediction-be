from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement="auto")
    email = Column(String(255), nullable=False, unique=True, index=True)
    password = Column(String(255), nullable=False)
    nama_lengkap = Column(String(255), nullable=False)
    nama_toko = Column(String(255), nullable=False, default="")
    role = Column(String(50), nullable=False)

class Sale(Base):
    __tablename__ = "sales"

    sale_id = Column(Integer, primary_key=True, autoincrement="auto")
    invoice = Column(String, nullable=True, index=True)
    tanggal_pembayaran = Column(DateTime, nullable=True, index=True)
    status_terakhir = Column(String(255), nullable=True, index=True)
    nama_produk = Column(String(255), nullable=True, index=True)
    jumlah_produk_dibeli = Column(Integer, nullable=True)
    harga_jual_idr = Column(Integer, nullable=True)
    total_penjualan_idr = Column(Integer, nullable=True)

    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False, index=True)