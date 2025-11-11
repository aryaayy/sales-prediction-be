from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

import pandas as pd
import numpy as np
from io import StringIO
import base64
from sqlalchemy.orm import Session
from database import get_db

import bcrypt
from jose import jwt, JWTError

import os
from dotenv import load_dotenv
import schemas, crud, models
from prediction import run_prediction

from datetime import datetime, timedelta, timezone

app = FastAPI(title="Sales Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081"],
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
UPLOAD_DIR = os.getenv("UPLOAD_DIR")

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
        role=user.role,
        csv_path=user.csv_path or ""
    )

@app.post("/api/user/register", tags=["User"], response_model=schemas.UserResponse)
async def create_user(user: schemas.UserCreate, conn: Session = Depends(get_db)):
    if crud.get_user_by_email(conn, user.email):
        raise HTTPException(409, "Email sudah digunakan")

    return crud.insert_user(conn, user)

@app.put("/api/user/edit_profile", tags=["User"], response_model=schemas.UserResponse)
async def edit_profile(user: schemas.UserEdit, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
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
        role=edited_user.role,
        csv_path=edited_user.csv_path or ""
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
        role=edited_user.role,
        csv_path=edited_user.csv_path or ""
    )

@app.delete("/api/user/{user_id}/delete", tags=["User"])
async def delete_account(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    db_user = crud.get_user_by_id(conn, user_id)

    if not db_user:
        raise HTTPException(400, "User tidak ditemukan")
    
    result = crud.delete_user(conn, db_user)

    return result

def hash_path(path: str) -> str:
    byte_path = path.encode('utf-8')
    hashed = bcrypt.hashpw(byte_path, bcrypt.gensalt())
    safe_hash = base64.urlsafe_b64encode(hashed).decode('utf-8').rstrip("=")
    return safe_hash

# DATASET
@app.post("/api/sales/upload", tags=["Dataset"], response_model=schemas.UserResponse)
async def upload_sales(file: UploadFile = File(...), conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)
    user = crud.get_user_by_email(conn, payload["email"])
    prev_csv = os.path.join(UPLOAD_DIR, user.csv_path)

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, detail="File format not supported")
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    try:
        csv_path = hash_path(f"user_id{user.user_id}_{file.filename}")
        csv_path += ".csv"

        with open(os.path.join(UPLOAD_DIR, csv_path), "wb") as f:
            contents = await file.read()
            f.write(contents)

        user.csv_path = csv_path

        conn.commit()
        conn.refresh(user)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(400, detail=f"Failed reading file: {str(e)}")
    
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("[()]", "", regex=True)
    df = df.replace({np.nan: None, pd.NaT: None})
    df = df.dropna(how="all")

    df["tanggal_pembayaran"] = pd.to_datetime(df["tanggal_pembayaran"], errors="coerce")
    df["user_id"] = user.user_id

    try:
        crud.delete_sales(conn, user.user_id)
        records = df.to_dict(orient="records")
        conn.bulk_insert_mappings(models.Sale, records)
        conn.commit()
        if prev_csv != f"{UPLOAD_DIR}None" and prev_csv != UPLOAD_DIR and os.path.exists(prev_csv):
            os.remove(prev_csv)
    except Exception as e:
        conn.rollback()
        raise HTTPException(500, detail=f"Database insertion failed: {str(e)}")
    
    return schemas.UserResponse(
        user_id=user.user_id,
        email=user.email,
        nama_lengkap=user.nama_lengkap,
        nama_toko=user.nama_toko,
        role=user.role,
        csv_path=user.csv_path
    )

@app.get("/api/sales/summary/{user_id}", tags=["Dataset"])
async def summarize_sales(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    sales = crud.get_sales_summary(conn, user_id)
    
    if sales == None:
        return None

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

    rows_count = crud.count_fetch_filter_sales(conn, user_id, year, status)

    return {
        "dataset": sales,
        "rows": rows_count.rows
    }

@app.get("/api/sales/products/{user_id}", tags=["Dataset"], response_model=list[schemas.ProductResponse])
async def my_products(user_id: int, limit: int = 10, offset: int = 0, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    result = crud.get_products(conn, user_id, limit, offset)

    products: schemas.ProductResponse = []

    number = offset + 1
    for row in result:
        products.append(schemas.ProductResponse(
            nomor_produk=number,
            nama_produk=row.nama_produk,
            total_transaksi=row.total_transaksi,
            total_penjualan=row.total_penjualan
        ))
        number += 1
    
    return products

# TOP PRODUCTS
@app.get("/api/top_products/{user_id}", tags=["Top Products"], response_model=list[schemas.TopProductsResponse])
async def my_top_products(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    result = crud.get_top_products(conn, user_id)

    top_products: schemas.TopProductsResponse = []

    for row in result:
        top_products.append(schemas.TopProductsResponse(
            nama_produk=row.nama_produk,
            total_unit_terjual=row.total_unit_terjual,
            total_transaksi=row.total_transaksi,
            total_penjualan=row.total_penjualan,
        ))

    return top_products

@app.get("/api/top_products/summary/{user_id}", tags=["Top Products"], response_model=schemas.TopProductsSummaryResponse)
async def my_top_products_summary(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    result = crud.get_top_products_summary(conn, user_id)

    return schemas.TopProductsSummaryResponse(
        total_penjualan_top=result.total_penjualan_top,
        total_unit_terjual_top=result.total_unit_terjual_top
    )

# STATISTICS
@app.get("/api/statistics/monthly_trend/{user_id}", tags=["Statistics"], response_model=list[schemas.SalesTrendResponse])
async def my_monthly_trend(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    result = list(crud.get_monthly_sales_trend(conn, user_id))

    trends: schemas.SalesTrendResponse = []

    is_first = True
    prev_total_penjualan = -1
    for row in reversed(result):
        difference = ((row.total_penjualan-prev_total_penjualan)/prev_total_penjualan)*100
        prev_total_penjualan = row.total_penjualan

        if is_first:
            is_first = False
            continue

        trends.append(schemas.SalesTrendResponse(
            bulan_pembayaran=row.bulan_pembayaran,
            tahun_pembayaran=row.tahun_pembayaran,
            total_penjualan=row.total_penjualan,
            pertumbuhan=difference
        ))

    return trends

@app.get("/api/statistics/transaction_analysis/{user_id}", tags=["Statistics"], response_model=schemas.TransactionAnalysisResponse)
async def my_transaction_analysis(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    result = crud.get_transaction_analysis(conn, user_id)

    return schemas.TransactionAnalysisResponse(
        avg_penjualan=result.avg_penjualan,
        max_penjualan=result.max_penjualan,
        min_penjualan=result.min_penjualan,
        median_penjualan=result.median_penjualan,
        std_penjualan=result.std_penjualan
    )

@app.get("/api/statistics/temporal_pattern/{user_id}", tags=["Statistics"], response_model=schemas.TemporalPatternResponse)
async def my_temporal_pattern(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    day = crud.get_temporal_day(conn, user_id)
    month = crud.get_temporal_month(conn, user_id)
    time_range = crud.get_temporal_time_range(conn, user_id)

    return schemas.TemporalPatternResponse(
        hari_transaksi=day.hari_transaksi,
        jumlah_transaksi_hari=day.jumlah_transaksi_hari,
        bulan_transaksi=month.bulan_transaksi,
        jumlah_transaksi_bulan=month.jumlah_transaksi_bulan,
        rentang_jam_transaksi=time_range.rentang_jam_transaksi,
        jumlah_transaksi_jam=time_range.jumlah_transaksi_jam,
    )

# PREDICTIONS
@app.post("/api/predictions/predict", tags=["Predictions"])
async def predict(args: schemas.PredictionArgs, background_tasks: BackgroundTasks, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    crud.delete_prediction_job(conn, args.user_id)
    crud.delete_all_predictions(conn, args.user_id)
    conn.commit()

    csv_path = f"uploads/sales/{args.csv_path}"
    background_tasks.add_task(run_prediction, csv_path, args.user_id)

    return {"status": "file executed successfully", "file": csv_path}

def get_job(user_id: int, conn: Session = Depends(get_db)):
    result = crud.get_prediction_job(conn, user_id)

    if result == None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction job not found."
        )

    return result

@app.get("/api/predictions/metrics/{user_id}", tags=["Predictions"])
async def my_prediction_metrics(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme), job = Depends(get_job)):
    payload = verify_token(token)

    if job.status == "running" or job.status == "failed":
        return {
            "job_status": job.status,
            "data": ""
        }

    result = crud.get_prediction_metrics(conn, user_id)

    metrics = schemas.PredictionMetricsResponse(
        prediction_metric_id=result.prediction_metric_id,
        arima_mae=result.arima_mae,
        arima_rmse=result.arima_rmse,
        arima_waktu_train=result.arima_waktu_train,
        arima_memori=result.arima_memori,
        lstm_mae=result.lstm_mae,
        lstm_rmse=result.lstm_rmse,
        lstm_waktu_train=result.lstm_waktu_train,
        lstm_memori=result.lstm_memori,
    )

    return {
        "job_status": job.status,
        "data": metrics
    }

@app.get("/api/predictions/comparisons/{user_id}", tags=["Predictions"])
async def my_prediction_comparisons(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme), job = Depends(get_job)):
    payload = verify_token(token)

    if job.status == "running" or job.status == "failed":
        return {
            "job_status": job.status,
            "data": []
        }

    result = crud.get_prediction_comparisons(conn, user_id)

    prediction_comparisons = []

    for row in result:
        prediction_comparisons.append(schemas.PredictionComparisonsResponse(
            prediction_comparison_id=row.prediction_comparison_id,
            hari=row.hari,
            hasil_total_penjualan_aktual=row.hasil_total_penjualan_aktual,
            hasil_total_penjualan_arima=row.hasil_total_penjualan_arima,
            hasil_total_penjualan_lstm=row.hasil_total_penjualan_lstm
        ))

    return {
        "job_status": job.status,
        "data": prediction_comparisons
    }

@app.get("/api/predictions/total/{user_id}", tags=["Predictions"])
async def my_total_predictions(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme), job = Depends(get_job)):
    payload = verify_token(token)

    if job.status == "running" or job.status == "failed":
        return {
            "job_status": job.status,
            "data": []
        }

    result = crud.get_total_predictions(conn, user_id)

    total_predictions = []

    for row in result:
        total_predictions.append(schemas.TotalPredictionResponse(
            total_prediction_id=row.total_prediction_id,
            hasil_tanggal=row.hasil_tanggal,
            hasil_total_penjualan_arima=row.hasil_total_penjualan_arima,
            hasil_total_penjualan_lstm=row.hasil_total_penjualan_lstm
        ))

    return {
        "job_status": job.status,
        "data": total_predictions
    }

@app.get("/api/predictions/daily/{user_id}", tags=["Predictions"])
async def my_daily_predicions(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme), job = Depends(get_job)):
    payload = verify_token(token)

    if job.status == "running" or job.status == "failed":
        return {
            "job_status": job.status,
            "data": []
        }

    result = crud.get_daily_predictions(conn, user_id)

    products = []

    for row in result:
        products.append(schemas.DailyPredictionsResponse(
            daily_product_prediction_id=row.daily_product_prediction_id,
            hari=row.hari,
            hasil_nama_produk_arima=row.hasil_nama_produk_arima,
            hasil_nama_produk_lstm=row.hasil_nama_produk_lstm,
        ))

    return {
        "job_status": job.status,
        "data": products
    }

@app.get("/api/predictions/weekly/{user_id}", tags=["Predictions"])
async def my_weekly_predicions(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme), job = Depends(get_job)):
    payload = verify_token(token)

    if job.status == "running" or job.status == "failed":
        return {
            "job_status": job.status,
            "data": []
        }

    result = crud.get_weekly_predictions(conn, user_id)

    products = []

    for row in result:
        products.append(schemas.WeeklyPredictionsResponse(
            weekly_product_prediction_id=row.weekly_product_prediction_id,
            minggu=row.minggu,
            hasil_nama_produk_arima=row.hasil_nama_produk_arima,
            hasil_nama_produk_lstm=row.hasil_nama_produk_lstm,
        ))

    return {
        "job_status": job.status,
        "data": products
    }

@app.get("/api/predictions/monthly/{user_id}", tags=["Predictions"])
async def my_monthly_predicions(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme), job = Depends(get_job)):
    payload = verify_token(token)

    if job.status == "running" or job.status == "failed":
        return {
            "job_status": job.status,
            "data": []
        }

    result = crud.get_monthly_predictions(conn, user_id)

    products = []

    for row in result:
        products.append(schemas.MonthlyPredictionsResponse(
            monthly_product_prediction_id=row.monthly_product_prediction_id,
            bulan=row.bulan,
            hasil_nama_produk_arima=row.hasil_nama_produk_arima,
            hasil_nama_produk_lstm=row.hasil_nama_produk_lstm,
        ))

    return {
        "job_status": job.status,
        "data": products
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
    expiration_time = datetime.now(timezone.utc) + timedelta(days=30)
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