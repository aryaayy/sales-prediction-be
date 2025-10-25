from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

import pandas as pd
import numpy as np
from io import StringIO
from sqlalchemy.orm import Session
from database import get_db

import bcrypt
from jose import jwt, JWTError

import os
from dotenv import load_dotenv
import schemas, crud, models


from datetime import datetime, timedelta, timezone

app = FastAPI(title="Sales Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"]
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# LOAD ENV VARIABLES
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
SALT = os.getenv("SALT").encode('utf-8')

# ROOT
@app.get("/api/", tags=["Root"])
async def index():
    return {"message": "Sales Predictor BE"}

# USER
@app.get("/api/user/me", tags=["User"], response_model=schemas.UserResponse)
async def read_user_me(conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    user = crud.get_user_by_email(conn, payload["email"])

    return schemas.UserResponse(
        user_id=user.user_id,
        email=user.email,
        nama_lengkap=user.nama_lengkap,
        nama_toko=user.nama_toko,
        role=user.role
    )

@app.post("/api/user/register", tags=["User"], response_model=schemas.UserResponse)
async def create_user(user: schemas.UserCreate, conn: Session = Depends(get_db)):
    if crud.get_user_by_email(conn, user.email):
        raise HTTPException(409, "Email sudah digunakan")

    return crud.insert_user(conn, user)

@app.put("/api/user/edit_profile", tags=["User"], response_model=schemas.UserResponse)
async def edit_profile(user: schemas.UserResponse, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    db_user = crud.get_user_by_id(conn, user.user_id)

    if not db_user:
        raise HTTPException(400, "User tidak ditemukan")

    edited_user = crud.update_user(conn, db_user, user)

    return schemas.UserResponse(
        user_id=edited_user.user_id,
        email=edited_user.email,
        nama_lengkap=edited_user.nama_lengkap,
        nama_toko=edited_user.nama_toko,
        role=edited_user.role
    )

@app.put("/api/user/change_password", tags=["User"], response_model=schemas.UserResponse)
async def change_password(user: schemas.UserChangePassword, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    db_user = crud.get_user_by_id(conn, user.user_id)

    if not db_user:
        raise HTTPException(400, "User tidak ditemukan")
    
    if db_user.password != hash_password(user.password):
        return HTTPException(401, "Password tidak cocok")

    edited_user = crud.update_password(conn, db_user, hash_password(user.new_password))

    return schemas.UserResponse(
        user_id=edited_user.user_id,
        email=edited_user.email,
        nama_lengkap=edited_user.nama_lengkap,
        nama_toko=edited_user.nama_toko,
        role=edited_user.role
    )

@app.delete("/api/user/{user_id}/delete", tags=["User"])
async def delete_account(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    db_user = crud.get_user_by_id(conn, user_id)

    if not db_user:
        raise HTTPException(400, "User tidak ditemukan")
    
    result = crud.delete_user(conn, db_user)

    return result

# DATASET
@app.post("/api/sales/upload", tags=["Dataset"])
async def upload_sales(file: UploadFile = File(...), conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, detail="File format not supported")

    try:
        contents = (await file.read()).decode("utf-8")
        df = pd.read_csv(StringIO(contents))
    except Exception as e:
        raise HTTPException(400, detail=f"Failed reading file: {str(e)}")
    
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("[()]", "", regex=True)
    df = df.replace({np.nan: None, pd.NaT: None})
    df = df.dropna(how="all")
    
    user = crud.get_user_by_email(conn, payload["email"])

    df["tanggal_pembayaran"] = pd.to_datetime(df["tanggal_pembayaran"], errors="coerce")
    df["user_id"] = user.user_id

    try:
        records = df.to_dict(orient="records")
        conn.bulk_insert_mappings(models.Sale, records)
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(500, detail=f"Database insertion failed: {str(e)}")
    
    return {"message": f"Successfully uploaded {len(df)} rows of sales data"}

@app.get("/api/sales/summary/{user_id}", tags=["Dataset"], response_model=schemas.DataSummaryResponse)
async def summarize_sales(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    sales = crud.get_sales_summary(conn, user_id)

    return schemas.DataSummaryResponse(
        total_transaksi=sales.total_transaksi,
        total_produk=sales.total_produk,
        periode_awal=sales.periode_awal,
        periode_akhir=sales.periode_akhir,
        total_status_selesai=sales.total_status_selesai,
        total_status_dibatalkan=sales.total_status_dibatalkan,
        total_status_dibatalkan_pembeli=sales.total_status_dibatalkan_pembeli,
        total_status_dibatalkan_penjual=sales.total_status_dibatalkan_penjual,
        total_status_dibatalkan_sistem=sales.total_status_dibatalkan_sistem,
        total_status_sedang_dikirim=sales.total_status_sedang_dikirim,
    )

@app.get("/api/sales/{user_id}", tags=["Dataset"])
async def my_sales(user_id: int, limit: int = 10, offset: int = 0, year: str = "all", status: str = "all", conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    sales_response = crud.get_sales(conn, user_id, limit, offset, year, status)

    sales: schemas.SalesDataResponse = []

    for row in sales_response:
        sales.append(schemas.SalesDataResponse(
            sale_id=row.sale_id,
            invoice=row.invoice,
            tanggal_pembayaran=row.tanggal_pembayaran,
            status_terakhir=row.status_terakhir,
            nama_produk=row.nama_produk,
            jumlah_produk_dibeli=row.jumlah_produk_dibeli,
            harga_jual_idr=row.harga_jual_idr,
            total_penjualan_idr=row.total_penjualan_idr
        ))

    rows_count = crud.count_fetch_filter(conn, user_id, year, status)

    return {
        "dataset": sales,
        "rows": rows_count.rows
    }

# AUTHENTICATION
def hash_password(password):
    byte_password = password.encode('utf-8')
    hashed = bcrypt.hashpw(byte_password, SALT).decode('utf-8')
    return hashed

def authenticate(credentials: schemas.UserLogin, conn: Session):
    user = crud.get_user_by_email(conn, credentials.email)
    if user:
        return hash_password(credentials.password) == user.password
    else:
        return False

def create_token(email: str):
    expiration_time = datetime.now(timezone.utc) + timedelta(hours=24)
    token = jwt.encode({"email": email, "exp": expiration_time.timestamp()}, SECRET_KEY, algorithm=ALGORITHM)
    return token

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        if not payload["email"]:
            raise HTTPException(401, "Invalid Access Token")

        return payload
    except JWTError:
        raise HTTPException(401, "Invalid Access Token")

@app.post("/token", tags=["Token"], response_model=schemas.Token)
async def token(form_data: OAuth2PasswordRequestForm = Depends(), conn: Session = Depends(get_db)):
    credentials = schemas.UserLogin(
        email=form_data.username,
        password=form_data.password
    )

    if not authenticate(credentials, conn):
        raise HTTPException(401, "Invalid Credentials!")
    
    access_token = create_token(credentials.email)

    return schemas.Token(
        access_token=access_token,
        token_type="bearer"
    )