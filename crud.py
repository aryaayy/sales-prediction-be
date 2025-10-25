from sqlalchemy import func, case, extract, not_, or_
from sqlalchemy.orm import Session
import models, schemas
from dotenv import load_dotenv
import bcrypt, os

# USERS
def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_user_by_id(db: Session, user_id: str):
    return db.query(models.User).filter(models.User.user_id == user_id).first()

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

def update_user(db: Session, db_user: models.User, user: schemas.UserResponse):
    db_user.email = user.email
    db_user.nama_lengkap = user.nama_lengkap
    db_user.nama_toko = user.nama_toko

    db.commit()
    db.refresh(db_user)

    return db_user

def update_password(db: Session, db_user: models.User, hashed_password: str):
    db_user.password = hashed_password

    db.commit()
    db.refresh(db_user)

    return db_user

def delete_user(db: Session, db_user: models.User):

    db.delete(db_user)
    db.commit()

    return {"message": "success"}

# DATASET
def get_sales(db: Session, user_id: int, limit: int, offset: int, year: str, status: str):
    if year == "all" and status == "all":
        return db.query(models.Sale).filter(models.Sale.user_id == user_id).limit(limit).offset(offset)
    
    if year == "all" and status != "others":
        return db.query(models.Sale).filter(models.Sale.user_id == user_id, models.Sale.status_terakhir == status).limit(limit).offset(offset)
    
    if year == "all" and status == "others":
        return db.query(models.Sale).filter(models.Sale.user_id == user_id, or_(not_(models.Sale.status_terakhir.in_(["Pesanan Selesai", "Dibatalkan Penjual", "Dibatalkan Pembeli", "Dibatalkan Sistem", "Sedang Dikirim"])), models.Sale.status_terakhir.is_(None))).limit(limit).offset(offset)
    
    if status == "all":
        return db.query(models.Sale).filter(models.Sale.user_id == user_id, extract("year", models.Sale.tanggal_pembayaran) == year).limit(limit).offset(offset)
    
    if status == "others":
        return db.query(models.Sale).filter(models.Sale.user_id == user_id, extract("year", models.Sale.tanggal_pembayaran) == year, or_(not_(models.Sale.status_terakhir.in_(["Pesanan Selesai", "Dibatalkan Penjual", "Dibatalkan Pembeli", "Dibatalkan Sistem", "Sedang Dikirim"])), models.Sale.status_terakhir.is_(None))).limit(limit).offset(offset)

    return db.query(models.Sale).filter(models.Sale.user_id == user_id, extract("year", models.Sale.tanggal_pembayaran) == year, models.Sale.status_terakhir == status).limit(limit).offset(offset)

def get_sales_summary(db: Session, user_id: int):
    result = db.query(
        func.count(models.Sale.sale_id).label("total_transaksi"),
        func.count(func.distinct(models.Sale.nama_produk)).label("total_produk"),
        func.sum(case((models.Sale.status_terakhir == "Pesanan Selesai", 1), else_=0)).label("total_status_selesai"),
        func.sum(case((models.Sale.status_terakhir.like("Dibatalkan%"), 1), else_=0)).label("total_status_dibatalkan"),
        func.sum(case((models.Sale.status_terakhir == "Dibatalkan Pembeli", 1), else_=0)).label("total_status_dibatalkan_pembeli"),
        func.sum(case((models.Sale.status_terakhir == "Dibatalkan Penjual", 1), else_=0)).label("total_status_dibatalkan_penjual"),
        func.sum(case((models.Sale.status_terakhir == "Dibatalkan Sistem", 1), else_=0)).label("total_status_dibatalkan_sistem"),
        func.sum(case((models.Sale.status_terakhir == "Sedang Dikirim", 1), else_=0)).label("total_status_sedang_dikirim"),
        func.min(models.Sale.tanggal_pembayaran).label("periode_awal"),
        func.max(models.Sale.tanggal_pembayaran).label("periode_akhir")
    ).filter(models.Sale.user_id == user_id).one()

    return result

def count_fetch_filter(db: Session, user_id: int, year: str, status: str):
    if year == "all" and status == "all":
        return db.query(func.count(models.Sale.sale_id).label("rows")).filter(models.Sale.user_id == user_id).one()
    
    if year == "all" and status != "others":
        return db.query(func.count(models.Sale.sale_id).label("rows")).filter(models.Sale.user_id == user_id, models.Sale.status_terakhir == status).one()
    
    if year == "all" and status == "others":
        return db.query(func.count(models.Sale.sale_id).label("rows")).filter(models.Sale.user_id == user_id, or_(not_(models.Sale.status_terakhir.in_(["Pesanan Selesai", "Dibatalkan Penjual", "Dibatalkan Pembeli", "Dibatalkan Sistem", "Sedang Dikirim"])), models.Sale.status_terakhir.is_(None))).one()
    
    if status == "all":
        return db.query(func.count(models.Sale.sale_id).label("rows")).filter(models.Sale.user_id == user_id, extract("year", models.Sale.tanggal_pembayaran) == year).one()
    
    if status == "others":
        return db.query(func.count(models.Sale.sale_id).label("rows")).filter(models.Sale.user_id == user_id, extract("year", models.Sale.tanggal_pembayaran) == year, or_(not_(models.Sale.status_terakhir.in_(["Pesanan Selesai", "Dibatalkan Penjual", "Dibatalkan Pembeli", "Dibatalkan Sistem", "Sedang Dikirim"])), models.Sale.status_terakhir.is_(None))).one()

    return db.query(func.count(models.Sale.sale_id).label("rows")).filter(models.Sale.user_id == user_id, extract("year", models.Sale.tanggal_pembayaran) == year, models.Sale.status_terakhir == status).one()