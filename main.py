from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from sqlalchemy.orm import Session
from database import get_db

import bcrypt
from jose import jwt, JWTError

import os
from dotenv import load_dotenv
import schemas
import crud

from datetime import datetime, timedelta, timezone

app = FastAPI(title="Sales Prediction", version="1.0.0")

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
    return {"message": "Sales Prediction BE"}

# USERS
@app.get("/api/user/me", tags=["Users"], response_model=schemas.UserResponse)
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

@app.post("/api/user/register", tags=["Users"], response_model=schemas.UserResponse)
async def create_user(user: schemas.UserCreate, conn: Session = Depends(get_db)):
    if crud.get_user_by_email(conn, user.email):
        raise HTTPException(409, "Email already in use")

    return crud.insert_user(conn, user)

@app.put("/api/user/edit_profile", response_model=schemas.UserResponse)
async def edit_profile(user: schemas.UserResponse, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    edited_user = crud.update_user(conn, user)

    return schemas.UserResponse(
        user_id=edited_user.user_id,
        email=edited_user.email,
        nama_lengkap=edited_user.nama_lengkap,
        nama_toko=edited_user.nama_toko,
        role=edited_user.role
    )

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