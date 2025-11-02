# ======================================================
# 🚀 Distributed Stock Price Prediction System (All-in-One)
# ======================================================
# Requirements:
# pip install fastapi uvicorn celery redis sqlalchemy yfinance prophet pandas
# Run:
#   1️⃣ Start Redis server  (docker run -d -p 6379:6379 redis)
#   2️⃣ Start worker:  celery -A main.celery worker --loglevel=info
#   3️⃣ Start API:     python main.py
# ======================================================

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from celery import Celery
from datetime import datetime
import yfinance as yf
from prophet import Prophet
import pandas as pd
import uvicorn

# ------------------------------------------------------
# Database setup (SQLite)
# ------------------------------------------------------
SQLALCHEMY_DATABASE_URL = "sqlite:///./stock_predictions.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class StockPrediction(Base):
    __tablename__ = "stock_predictions"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    predicted_price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ------------------------------------------------------
# Celery setup (Redis backend)
# ------------------------------------------------------
celery = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

@celery.task
def run_prediction(symbol: str):
    print(f"🔍 Fetching stock data for {symbol}...")
    df = yf.download(symbol, period="1y")
    df = df.reset_index()[["Date", "Close"]]
    df.columns = ["ds", "y"]

    print("📈 Training Prophet model...")
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    predicted_price = forecast["yhat"].iloc[-1]

    db = SessionLocal()
    record = StockPrediction(
        symbol=symbol,
        predicted_price=float(predicted_price),
        created_at=datetime.utcnow()
    )
    db.add(record)
    db.commit()
    db.close()

    print(f"✅ Prediction completed for {symbol}: ${predicted_price:.2f}")
    return predicted_price

# ------------------------------------------------------
# FastAPI setup
# ------------------------------------------------------
app = FastAPI(title="Distributed Stock Price Prediction System")

class PredictionRequest(BaseModel):
    symbol: str

class PredictionResponse(BaseModel):
    symbol: str
    predicted_price: float | None = None
    created_at: str | None = None
    message: str | None = None

    class Config:
        orm_mode = True

@app.post("/predict", response_model=PredictionResponse)
def predict_stock(data: PredictionRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_prediction.delay, data.symbol)
    return {"message": f"Prediction started for {data.symbol}", "symbol": data.symbol}

@app.get("/predictions", response_model=list[PredictionResponse])
def get_predictions():
    db = SessionLocal()
    preds = db.query(StockPrediction).all()
    db.close()
    return preds

# ------------------------------------------------------
# Main entry point
# ------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting FastAPI server on http://localhost:8000 ...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
