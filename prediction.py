from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

import models

load_dotenv()
host=os.getenv("DB_HOST")
port=os.getenv("DB_PORT")
user=os.getenv("DB_USER")
password=os.getenv("DB_PASS")
database=os.getenv("DB_NAME")

DB_URL = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"

engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# -*- coding: utf-8 -*-

"""Data Manipulation & Processing"""

import pandas as pd
import numpy as np

"""Data Visualization"""

import matplotlib.pyplot as plt
import seaborn as sns

"""Time Series Analysis & Forecasting"""

from statsmodels.tsa.arima.model import ARIMA

"""Model Evaluation Metrics"""

import tensorflow as tf
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

"""LSTM Model"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

"""Data Preprocessing & Normalization"""

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import scipy.stats as stats

"""Mathematics & Statistical Operations"""

import math

"""Calendar"""

import calendar
import time

"""## 2. Data Loading

Mengambil data dari Google Sheets dan mengonversinya menjadi DataFrame Pandas
"""
def run_prediction(csv_path: str, user_id: int):
    db = SessionLocal()

    try:
        job = models.PredictionJob(
            status='running',
            user_id=user_id
        )

        db.add(job)
        db.commit()

        df = pd.read_csv(csv_path)

        df['Tanggal Pembayaran'] = pd.to_datetime(df['Tanggal Pembayaran'], errors='coerce')

        df['Year'] = df['Tanggal Pembayaran'].dt.year
        df['Month'] = df['Tanggal Pembayaran'].dt.month
        df['Day'] = df['Tanggal Pembayaran'].dt.day
        df = df.dropna(subset=['Month'])

        df['Z_Score_Total_Penjualan'] = zscore(df['Total Penjualan (IDR)'])
        df['Z_Score_Harga_Jual'] = zscore(df['Harga Jual (IDR)'])

        outliers_nominal = df[df['Z_Score_Total_Penjualan'].abs() > 3]
        outliers_balance = df[df['Z_Score_Harga_Jual'].abs() > 3]

        transactions_per_year = df.groupby('Year').size()
        transactions_per_month = df.groupby(['Year', 'Month']).size()
        transactions_per_day = df.groupby('Tanggal Pembayaran').size()

        most_transacted_day = transactions_per_day.idxmax()
        least_transacted_day = transactions_per_day.idxmin()

        balance_trend = df.groupby('Tanggal Pembayaran')['Harga Jual (IDR)'].last()

        status_counts = df['Status Terakhir'].value_counts()

        status_description = {
            'Pesanan Selesai': 'Pesanan berhasil diselesaikan',
            'Dibatalkan Sistem': 'Pesanan dibatalkan oleh sistem',
            'Dibatalkan Penjual': 'Pesanan dibatalkan oleh penjual',
            'Dibatalkan Pembeli': 'Pesanan dibatalkan oleh pembeli'
        }

        status_percent = (status_counts / status_counts.sum()) * 100

        total_transactions = df.groupby('Status Terakhir')['Total Penjualan (IDR)'].sum()

        average_transactions = df.groupby('Status Terakhir')['Total Penjualan (IDR)'].mean()

        total_transactions_formatted = total_transactions.apply(lambda x: f"{x:,.0f}")  # Format dengan pemisah ribuan

        status_counts = df['Status Terakhir'].value_counts()

        aligned_status_counts = status_counts.reindex(average_transactions.index)

        df_top_sales = df.sort_values(by='Total Penjualan (IDR)', ascending=False).head(10)

        df['Tanggal Pembayaran'] = pd.to_datetime(df['Tanggal Pembayaran'])
        df['Month'] = df['Tanggal Pembayaran'].dt.to_period('M')

        daily_balance = df.groupby('Tanggal Pembayaran')['Total Penjualan (IDR)'].mean()

        daily_mean_balance = daily_balance.mean()
        daily_formatted_balance = f"Rp{daily_mean_balance:,.0f}".replace(',', '.')

        monthly_balance = df.groupby('Month')['Total Penjualan (IDR)'].mean()

        monthly_mean_balance = monthly_balance.mean()
        monthly_formatted_balance = f"Rp{monthly_mean_balance:,.0f}".replace(',', '.')

        df['Tanggal Pembayaran'] = pd.to_datetime(df['Tanggal Pembayaran'], errors='coerce')

        df['Month'] = df['Tanggal Pembayaran'].dt.month
        monthly_sales = df.groupby('Month')['Total Penjualan (IDR)'].sum()

        monthly_sales.index = monthly_sales.index.map(lambda x: calendar.month_name[x])

        df['Tanggal Pembayaran'] = pd.to_datetime(df['Tanggal Pembayaran'], errors='coerce')

        df['DayOfWeek'] = df['Tanggal Pembayaran'].dt.dayofweek  # 0=Senin, 6=Minggu

        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        weekly_sales = df.groupby('DayOfWeek')['Total Penjualan (IDR)'].sum()

        data_sales = df.groupby('Tanggal Pembayaran')['Total Penjualan (IDR)'].sum().values.reshape(-1, 1)

        scaler_sales = MinMaxScaler(feature_range=(0, 1))
        scaled_sales = scaler_sales.fit_transform(data_sales)

        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)

        time_step = 30

        X_sales, y_sales = create_dataset(scaled_sales, time_step)

        train_size = int(len(X_sales) * 0.8)
        X_train_sales, X_test_sales = X_sales[:train_size], X_sales[train_size:]
        y_train_sales, y_test_sales = y_sales[:train_size], y_sales[train_size:]

        X_train_sales = X_train_sales.reshape(X_train_sales.shape[0], X_train_sales.shape[1], 1)
        X_test_sales = X_test_sales.reshape(X_test_sales.shape[0], X_test_sales.shape[1], 1)

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        import tracemalloc

        tracemalloc.start()

        model = Sequential()
        model.add(Input(shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))

        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=tf.keras.optimizers.SGD(learning_rate=1.0000e-04, momentum=0.9),
            metrics=[tf.keras.metrics.MeanAbsoluteError,
                    tf.keras.metrics.RootMeanSquaredError]
        )

        start_lstm = time.time()
        history = model.fit(X_train_sales, y_train_sales, epochs=100, batch_size=32, verbose=1)
        end_lstm = time.time()

        current_lstm, peak_lstm = tracemalloc.get_traced_memory()

        tracemalloc.stop()

        # df_arima = df.groupby('Tanggal Pembayaran')['Total Penjualan (IDR)'].sum()

        # df_arima = df_arima.asfreq('D')

        df_arima = df
        df_arima['Tanggal Pembayaran'] = df_arima['Tanggal Pembayaran'].dt.date
        df_arima = df_arima.groupby('Tanggal Pembayaran')['Total Penjualan (IDR)'].sum()

        from statsmodels.tsa.stattools import adfuller

        # Fill NaN values with 0
        df_arima = df_arima.fillna(0)

        result = adfuller(df_arima)

        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        import matplotlib.pyplot as plt

        from statsmodels.tsa.arima.model import ARIMA

        tracemalloc.start()

        arima_model = ARIMA(df_arima, order=(1, 0, 1), trend='c')

        start_arima = time.time()
        arima_fit = arima_model.fit()
        end_arima = time.time()

        current_arima, peak_arima = tracemalloc.get_traced_memory()

        tracemalloc.stop()

        # Jika ingin di level (tanpa log), set IS_LOG=False dan ganti y_fit=df_arima
        # IS_LOG = True  # <-- setel True bila memakai log1p
        # y_fit = np.log1p(df_arima.clip(lower=0)) if IS_LOG else df_arima

        # from statsmodels.tsa.arima.model import ARIMA
        # arima_model = ARIMA(y_fit, order=(1, 0, 1), trend='c')  # intercept agar level tidak drop
        # arima_fit = arima_model.fit()

        # # === FORECAST 90 HARI ===
        # steps = 90
        # forecast_index = pd.date_range(df_arima.index[-1] + pd.Timedelta(days=1),
        #                             periods=steps, freq=(df_arima.index.freq or 'D'))

        # fc = arima_fit.forecast(steps=steps)
        # if IS_LOG:
        #     fc = np.expm1(fc)  # inverse dari log1p
        # forecast_arima = pd.Series(np.asarray(fc).reshape(-1), index=forecast_index).clip(lower=0)

        history_loss = history.history['loss']
        history_mae = history.history['mean_absolute_error']
        history_rmse = history.history['root_mean_squared_error']

        test_loss, test_mae, test_rmse = model.evaluate(X_test_sales, y_test_sales, verbose=1)

        """Menampilkan hasil evaluasi"""

        results = {
            "Test Loss": [test_loss],
            "Test MAE": [test_mae],
            "Test RMSE": [test_rmse],
        }

        results_df = pd.DataFrame(results)

        y_pred_sales = model.predict(X_test_sales)

        # INI PREDIKSI LSTM
        def predict_future_sales(model, scaler, data, time_step, days):
            last_data = data[-time_step:].reshape(1, time_step, 1)
            predicted_sales = []
            for _ in range(days):
                pred_sales = model.predict(last_data)
                predicted_sales.append(pred_sales[0, 0])
                last_data = np.append(last_data[:, 1:, :], pred_sales.reshape(1, 1, 1), axis=1)

            predicted_sales = scaler.inverse_transform(np.array(predicted_sales).reshape(-1, 1))
            return predicted_sales


        predicted_sales = predict_future_sales(model, scaler_sales, scaled_sales, time_step, days=90)

        predicted_sales_df = pd.DataFrame(predicted_sales, columns=["Predicted Sales"])
        predicted_sales_df['Day'] = predicted_sales_df.index + 1
        predicted_sales_df.set_index('Day', inplace=True)

        predicted_sales_df = pd.DataFrame(predicted_sales, columns=["Predicted Sales"])
        predicted_sales_df['Day'] = predicted_sales_df.index + 1


        predicted_sales_df['Day'] = predicted_sales_df.index + 1
        predicted_sales_df.set_index('Day', inplace=True)

        predicted_sales_df['Predicted Sales (IDR)'] = predicted_sales_df['Predicted Sales'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.'))
        predicted_sales_df.drop(columns=['Predicted Sales'], inplace=True)

        predicted_sales_df = pd.DataFrame(predicted_sales, columns=["Predicted Sales"])
        predicted_sales_df['Day'] = predicted_sales_df.index + 1
        predicted_sales_df.set_index('Day', inplace=True)
        predicted_sales_df['Predicted Sales (IDR)'] = predicted_sales_df['Predicted Sales'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.'))
        predicted_sales_df.drop(columns=['Predicted Sales'], inplace=True)

        print("INI LSTM")
        print(predicted_sales)
        print(predicted_sales_df)

        from sklearn.metrics import mean_absolute_error, mean_squared_error

        pred_last30 = arima_fit.get_prediction(start=len(df_arima)-30, end=len(df_arima)-1)
        y_pred_last30 = np.clip(pred_last30.predicted_mean.values, 0, None)

        arima_mae = mean_absolute_error(df_arima[-30:], y_pred_last30)
        arima_rmse = np.sqrt(mean_squared_error(df_arima[-30:], y_pred_last30))

        results = {
            "Test Loss": [test_loss],  # Jika ada evaluasi lainnya seperti loss
            "Test MAE": [arima_mae],  # Hasil MAE
            "Test RMSE": [arima_rmse],  # Hasil RMSE
        }

        results_df = pd.DataFrame(results)

        # residuals = df_arima[-30:].values - forecast_arima[:30]

        df_arima = df_arima.asfreq('D')
        df_arima = df_arima.fillna(0)

        # INI PREDIKSI ARIMA
        def predict_future_arima(model, data, steps):
            idx = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=steps, freq=(data.index.freq or 'D'))
            fc = model.forecast(steps=steps)
            fc = np.clip(np.asarray(fc).reshape(-1), 0, None)
            return pd.Series(fc, index=idx)


        forecast_arima_future = predict_future_arima(arima_fit, df_arima, steps=90)
        print("INI ARIMA")
        print(forecast_arima_future)

        df['Tanggal Pembayaran'] = pd.to_datetime(df['Tanggal Pembayaran'], errors='coerce')

        produk_terjual_unik = (
            df.loc[df['Total Penjualan (IDR)'] > 0, 'Nama Produk']
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        produk_per_hari = (
            df[df['Total Penjualan (IDR)'] > 0]
            .groupby(df['Tanggal Pembayaran'].dt.date)['Nama Produk']
            .unique()
            .apply(list)
        )

        pd.Series(produk_terjual_unik, name="Nama Produk").to_csv("produk_terjual_unik.csv", index=False)
        produk_per_hari.to_csv("produk_terjual_per_hari.csv")

        rev_per_produk_harian = (
            df.groupby(['Tanggal Pembayaran','Nama Produk'])['Total Penjualan (IDR)']
            .sum()
            .unstack(fill_value=0)
            .sort_index()
        )

        window_hari = 60
        cutoff_date = rev_per_produk_harian.index.max() - pd.Timedelta(days=window_hari-1)
        recent_rev = rev_per_produk_harian.loc[rev_per_produk_harian.index >= cutoff_date]
        if recent_rev.empty:
            recent_rev = rev_per_produk_harian.copy()

        total_rev_recent_per_produk = recent_rev.sum(axis=0)
        epsilon = 1e-9
        total_recent_all = total_rev_recent_per_produk.sum()
        produk_share = (total_rev_recent_per_produk / (total_recent_all + epsilon)).fillna(0)

        top_k = 30
        top_produk = produk_share.sort_values(ascending=False).head(top_k).index.tolist()
        share_top = produk_share[top_produk].copy()
        share_lainnya = 1.0 - share_top.sum()
        share_top['__Lainnya__'] = max(0.0, share_lainnya)

        def allocate_per_product(total_forecast_vector, share_series):
            """Alokasikan total IDR harian → IDR per produk harian."""
            alloc = np.outer(total_forecast_vector, share_series.values)
            out_df = pd.DataFrame(alloc, columns=share_series.index)
            out_df.index = np.arange(1, len(total_forecast_vector)+1)  # Day 1..H
            return out_df

        def top_produk_per_hari(pred_df, N=10):
            """Ambil nama Top-N produk (IDR) untuk tiap hari."""
            top_list = {}
            for day, row in pred_df.iterrows():
                row_riil = row.drop(labels='__Lainnya__', errors='ignore')
                top_list[day] = row_riil.sort_values(ascending=False).head(N).index.tolist()
            return top_list

        def top_produk_agregat_30hari(pred_df, N=20):
            """Top-N produk berdasarkan total 30 hari (IDR)."""
            cols = [c for c in pred_df.columns if c != '__Lainnya__']
            total_30 = pred_df[cols].sum(axis=0).sort_values(ascending=False)
            return total_30.head(N)

        pred_per_produk_lstm = None
        try:
            forecast_lstm_idr = np.asarray(predicted_sales).reshape(-1)
            pred_per_produk_lstm = allocate_per_product(forecast_lstm_idr, share_top)

            N_harian = 30  # <- jumlah nama produk per hari
            top10_harian_lstm = top_produk_per_hari(pred_per_produk_lstm, N=N_harian)
            top20_agregat_lstm = top_produk_agregat_30hari(pred_per_produk_lstm, N=20)

            print("\n[Prediksi LSTM] Top produk per hari (Day 1..5):")
            for d in range(1, min(6, len(top10_harian_lstm)+1)):
                print(f"Day {d}: {top10_harian_lstm[d]}")

            print("\n[Prediksi LSTM] TOP-20 Produk Agregat 30 Hari (IDR):")
            print(top20_agregat_lstm.apply(lambda x: f"Rp{x:,.0f}".replace(',', '.')))

        except NameError as e:
            print("\n[Prediksi LSTM] Error:", e)
            print("Pastikan variabel 'predicted_sales' dan 'share_top' sudah terdefinisi.")

        """Prediksi berdasarkan ARIMA"""

        pred_per_produk_arima = None
        try:
            # 1) Normalisasi share hanya jika perlu
            if not np.isclose(float(share_top.sum()), 1.0):
                share_top = (share_top / max(float(share_top.sum()), 1e-9)).copy()

            # 2) Selalu bersihkan & jepit forecast (WAJIB di dalam try dan di luar if di atas)
            fa = np.asarray(forecast_arima.values, dtype=float).reshape(-1)

            # Jangan nol-kan Inf/NaN diam-diam; pakai fallback yang masuk akal
            if not np.isfinite(fa).all():
                # Ambil median 90 hari TERAMATI terakhir sebagai fallback
                recent_idx = df_arima.index[observed_mask] if 'observed_mask' in locals() else df_arima.index
                if len(recent_idx) > 0:
                    cut_90 = recent_idx.max() - pd.Timedelta(days=89)
                    recent_90 = df_arima.loc[(df_arima.index >= cut_90) & (observed_mask if 'observed_mask' in locals() else True)]
                    recent_90 = recent_90.replace(0, np.nan)
                    fb = float(np.nanmedian(recent_90))
                    if not np.isfinite(fb):
                        fb = float(np.nanmedian(df_arima.replace(0, np.nan)))
                else:
                    fb = float(np.nanmedian(df_arima.replace(0, np.nan)))
                fa = np.where(np.isfinite(fa), fa, fb)

            # Clamp agar tidak meledak: 0 .. 10×p90 historis (hari teramati)
            hist_obs = (df_arima.loc[observed_mask] if 'observed_mask' in locals() else df_arima).replace(0, np.nan).dropna()
            if len(hist_obs) == 0:
                hist_obs = df_arima.replace(0, np.nan).dropna()
            p90 = float(np.nanpercentile(hist_obs, 90)) if len(hist_obs) else 1.0
            upper_cap = max(10.0 * p90, 1.0)

            fa = np.clip(fa, 0, upper_cap)   # jepit atas-bawah
            forecast_arima_idr = fa          # siap untuk alokasi

            # 3) Alokasi produk
            pred_per_produk_arima = allocate_per_product(forecast_arima_idr, share_top)

            # 4) Ambil Top-N
            N_harian = 10
            top10_harian_arima = top_produk_per_hari(pred_per_produk_arima, N=N_harian)
            top20_agregat_arima = top_produk_agregat_30hari(pred_per_produk_arima, N=20)

            print("\n[Prediksi ARIMA] Top-10 produk per hari (Day 1..5):")
            for d in range(1, min(6, len(top10_harian_arima)+1)):
                print(f"Day {d}: {top10_harian_arima[d]}")

            print("\n[Prediksi ARIMA] TOP-20 Produk Agregat 30 Hari (IDR):")
            print(top20_agregat_arima.apply(lambda x: f"Rp{x:,.0f}".replace(',', '.')))

        except NameError as e:
            print("\n[Prediksi ARIMA] Error:", e)
            print("Pastikan variabel 'forecast_arima' dan 'share_top' sudah ada.")
        except Exception as e:
            print("\n[Prediksi ARIMA] Error umum:", repr(e))

            # --- Fungsi tambahan untuk agregasi mingguan dan bulanan ---
        def aggregate_to_weeks(pred_df, days_per_week=7):
            """Gabungkan prediksi harian menjadi mingguan (mean total IDR per minggu)."""
            n_weeks = len(pred_df) // days_per_week
            weekly_agg = []
            for i in range(n_weeks):
                chunk = pred_df.iloc[i * days_per_week:(i + 1) * days_per_week]
                weekly_agg.append(chunk.sum(axis=0))
            weekly_df = pd.DataFrame(weekly_agg)
            weekly_df.index = [f"Week {i+1}" for i in range(n_weeks)]
            return weekly_df

        def aggregate_to_months(pred_df, days_per_month=30):
            """Gabungkan prediksi harian menjadi bulanan."""
            n_months = len(pred_df) // days_per_month
            monthly_agg = []
            for i in range(n_months):
                chunk = pred_df.iloc[i * days_per_month:(i + 1) * days_per_month]
                monthly_agg.append(chunk.sum(axis=0))
            monthly_df = pd.DataFrame(monthly_agg)
            monthly_df.index = [f"Month {i+1}" for i in range(n_months)]
            return monthly_df

        # --- FUNGSI TAMBAHAN: ambil top produk per minggu/bulan ---
        def top_produk_per_periode(pred_df, N=10):
            """Ambil top-N produk per periode (minggu/bulan)."""
            top_dict = {}
            for idx, row in pred_df.iterrows():
                row_riil = row.drop(labels='__Lainnya__', errors='ignore')
                top_dict[idx] = row_riil.sort_values(ascending=False).head(N).index.tolist()
            return top_dict


        # ===========================
        #  PREDIKSI LSTM (7/28/90)
        # ===========================
        print("\n[Prediksi LSTM] ====")
        forecast_lstm_idr = np.asarray(predicted_sales).reshape(-1)
        pred_per_produk_lstm = allocate_per_product(forecast_lstm_idr, share_top)

        # ---- Harian (7 hari) ----
        pred_7hari_lstm = pred_per_produk_lstm.head(7)
        top7harian_lstm = top_produk_per_hari(pred_7hari_lstm, N=5)

        # ---- Mingguan (4 minggu) ----
        weekly_lstm = aggregate_to_weeks(pred_per_produk_lstm.head(28), days_per_week=7)
        top4mingguan_lstm = top_produk_per_periode(weekly_lstm, N=5)

        # ---- Bulanan (3 bulan) ----
        monthly_lstm = aggregate_to_months(pred_per_produk_lstm.head(90), days_per_month=30)
        top3bulanan_lstm = top_produk_per_periode(monthly_lstm, N=5)

        # Cetak hasil
        print("\n[LSTM] Top produk harian 7 hari ke depan:")
        for day, prods in top7harian_lstm.items():
            print(f"Day {day}: {prods}")

        print("\n[LSTM] Top produk mingguan 4 minggu ke depan:")
        for week, prods in top4mingguan_lstm.items():
            print(f"{week}: {prods}")

        print("\n[LSTM] Top produk bulanan 3 bulan ke depan:")
        for month, prods in top3bulanan_lstm.items():
            print(f"{month}: {prods}")


        # ===========================
        #  PREDIKSI ARIMA (7/28/90)
        # ===========================
        print("\n[Prediksi ARIMA] ====")
        forecast_arima_idr = np.asarray(forecast_arima_future).reshape(-1)
        pred_per_produk_arima = allocate_per_product(forecast_arima_idr, share_top)

        # ---- Harian (7 hari) ----
        pred_7hari_arima = pred_per_produk_arima.head(7)
        top7harian_arima = top_produk_per_hari(pred_7hari_arima, N=5)

        # ---- Mingguan (4 minggu) ----
        weekly_arima = aggregate_to_weeks(pred_per_produk_arima.head(28), days_per_week=7)
        top4mingguan_arima = top_produk_per_periode(weekly_arima, N=5)

        # ---- Bulanan (3 bulan) ----
        monthly_arima = aggregate_to_months(pred_per_produk_arima.head(90), days_per_month=30)
        top3bulanan_arima = top_produk_per_periode(monthly_arima, N=5)

        print("\n[ARIMA] Top produk harian 7 hari ke depan:")
        for day, prods in top7harian_arima.items():
            print(f"Day {day}: {prods}")

        print("\n[ARIMA] Top produk mingguan 4 minggu ke depan:")
        for week, prods in top4mingguan_arima.items():
            print(f"{week}: {prods}")

        print("\n[ARIMA] Top produk bulanan 3 bulan ke depan:")
        for month, prods in top3bulanan_arima.items():
            print(f"{month}: {prods}")

        # COMPARISON
        actual_last_90 = df_arima[-90:].fillna(0)

        # Buat index prediksi
        arima_index = forecast_arima_future.index
        lstm_index = pd.date_range(start=actual_last_90.index[-1] + pd.Timedelta(days=1), periods=90, freq='D')

        # Konversi hasil prediksi ke Series agar sejajar
        lstm_series = pd.Series(np.array(predicted_sales).reshape(-1), index=lstm_index)
        arima_series = pd.Series(forecast_arima_future.values.reshape(-1), index=arima_index)

        # Buat DataFrame perbandingan ringkas
        comparison_result = {
            "hasil_aktual_l10": actual_last_90.tail(10).values,
            "hasil_arima_f10": arima_series.head(10).values,
            "hasil_lstm_f10": lstm_series.head(10).values
        }

        print("\n=== PERBANDINGAN DATA AKTUAL VS ARIMA VS LSTM ===")
        print(comparison_result)

        # --- RETURN TAMBAHAN DALAM JSON ---
        result = {
            "arima_mae": arima_mae,
            "arima_rmse": arima_rmse,
            "arima_memori": round(peak_arima / (1024 ** 2), 2),
            "arima_waktu_train": end_arima-start_arima,
            "lstm_mae": test_mae,
            "lstm_rmse": test_rmse,
            "lstm_memori": round(peak_lstm / (1024 ** 2), 2),
            "lstm_waktu_train": end_lstm-start_lstm,
            "total_arima": forecast_arima_future,
            "total_lstm": predicted_sales,
            "top_7hari_arima": top7harian_arima,
            "top_4minggu_arima": top4mingguan_arima,
            "top_3bulan_arima": top3bulanan_arima,
            "top_7hari_lstm": top7harian_lstm,
            "top_4minggu_lstm": top4mingguan_lstm,
            "top_3bulan_lstm": top3bulanan_lstm,
        }

        # insert db

        prediction_metric = models.PredictionMetric(
            arima_mae = result["arima_mae"],
            arima_rmse = result["arima_rmse"],
            arima_waktu_train = result["arima_waktu_train"],
            arima_memori = result["arima_memori"],
            lstm_mae = result["lstm_mae"],
            lstm_rmse = result["lstm_rmse"],
            lstm_waktu_train = result["lstm_waktu_train"],
            lstm_memori = result["lstm_memori"],
            user_id = user_id,
        )

        # Convert LSTM array into DataFrame with same index as arima_series
        lstm_df = pd.DataFrame(result["total_lstm"].flatten(), index=result["total_arima"].index, columns=["hasil_total_penjualan_lstm"])

        # Convert ARIMA series into DataFrame
        arima_df = pd.DataFrame(result["total_arima"], columns=["hasil_total_penjualan_arima"])

        # Combine both
        merged_df = pd.concat([arima_df, lstm_df], axis=1).reset_index()
        merged_df.rename(columns={"index": "hasil_tanggal"}, inplace=True)

        total_predictions = [
            models.TotalPrediction(
                hasil_tanggal=row.hasil_tanggal,
                hasil_total_penjualan_arima=row.hasil_total_penjualan_arima,
                hasil_total_penjualan_lstm=row.hasil_total_penjualan_lstm,
                user_id=user_id
            )
            for _, row in merged_df.iterrows()
        ]

        daily_predictions = []

        for day, products_arima in result["top_7hari_arima"].items():
            products_lstm = result["top_7hari_lstm"][day]
            for product_arima, product_lstm in zip(products_arima, products_lstm):
                daily_predictions.append(
                    models.DailyProductPrediction(
                        hari = day,
                        hasil_nama_produk_arima = product_arima,
                        hasil_nama_produk_lstm = product_lstm,
                        user_id = user_id,
                    )
                )

        weekly_predictions = []

        for week, products_arima in result["top_4minggu_arima"].items():
            products_lstm = result["top_4minggu_lstm"][week]
            week_int = int(''.join(filter(str.isdigit, str(week))))
            for product_arima, product_lstm in zip(products_arima, products_lstm):
                weekly_predictions.append(
                    models.WeeklyProductPrediction(
                        minggu = week_int,
                        hasil_nama_produk_arima = product_arima,
                        hasil_nama_produk_lstm = product_lstm,
                        user_id = user_id,
                    )
                )

        monthly_predictions = []

        for month, products_arima in result["top_3bulan_arima"].items():
            products_lstm = result["top_3bulan_lstm"][month]
            month_int = int(''.join(filter(str.isdigit, str(month))))
            for product_arima, product_lstm in zip(products_arima, products_lstm):
                monthly_predictions.append(
                    models.MonthlyProductPrediction(
                        bulan = month_int,
                        hasil_nama_produk_arima = product_arima,
                        hasil_nama_produk_lstm = product_lstm,
                        user_id = user_id,
                    )
                )

        prediction_comparisons = []

        for i, (actual, arima, lstm) in enumerate(zip(comparison_result["hasil_aktual_l10"], comparison_result["hasil_arima_f10"], comparison_result["hasil_lstm_f10"]), start=1):
            prediction_comparisons.append(
                models.PredictionComparison(
                    hari=i,
                    hasil_total_penjualan_aktual=actual,
                    hasil_total_penjualan_arima=arima,
                    hasil_total_penjualan_lstm=lstm,
                    user_id=user_id
                )
            )

        db.add(prediction_metric)
        db.add_all(prediction_comparisons)
        db.add_all(total_predictions)
        db.add_all(daily_predictions)
        db.add_all(weekly_predictions)
        db.add_all(monthly_predictions)

        db.commit()
    except Exception as e:
        db.rollback()
        job.status = 'failed'
        db.commit()
        print("failed: ", e)
    finally:
        job.status = 'success'
        db.commit()
        db.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        sys.exit(1)
    
    csv_path = sys.argv[1]
    user_id = int(sys.argv[2])
    # with open(csv_path, "r") as f:
    #     contents = f.read()
    #     print(contents)
    run_prediction(csv_path)
    # result = run_prediction("D:/Arya/Project/lala/data/tokped.csv")
    # result = run_prediction("../uploads/sales/JDJiJDEyJGMvZGtJNlB2ZVczLmpPQlVmZS5hN09ENFp5M0IuWllGWGxnemJkSFRkYVdvUjdtR0xQcUZp.csv")
    