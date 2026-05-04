# # main.py — EV Battery SOH Predictor (High-Concurrency Optimized)

# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# import asyncio
# import numpy as np
# import pandas as pd
# from tensorflow import keras
# from sklearn.preprocessing import MinMaxScaler
# import io
# import os

# from agent import battery_agent, BatteryState
# from fastapi import WebSocket, WebSocketDisconnect
# from websocket_consumer import read_stream_and_predict

# from dotenv import load_dotenv
# load_dotenv()


# app = FastAPI(title="EV Battery SOH Predictor")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Model load karo startup par
# model = keras.models.load_model("best_tcn_v2.keras")

# FEATURES    = ["Capacity", "Re", "Rct", "ambient_temperature"]
# WINDOW_SIZE = 50

# @app.get("/")
# def root():
#     return {"message": "EV Battery SOH Prediction API", "status": "running"}

# @app.post("/predict")
# async def predict_soh(file: UploadFile = File(...)):
#     # CSV read karo (encoding safe)
#     contents = await file.read()
#     try:
#         df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
#     except UnicodeDecodeError:
#         df = pd.read_csv(io.StringIO(contents.decode("latin-1")))

#     # Validate karo
#     missing = [col for col in FEATURES if col not in df.columns]
#     if missing:
#         return {"error": f"Missing columns: {missing}"}

#     if len(df) < WINDOW_SIZE + 1:
#         return {"error": f"Need at least {WINDOW_SIZE + 1} rows"}

#     # Scale karo
#     scaler = MinMaxScaler()
#     df_scaled = df.copy()
#     df_scaled[FEATURES] = scaler.fit_transform(df[FEATURES])

#     # Window banao
#     feature_data = df_scaled[FEATURES].values
#     windows = []
#     for i in range(len(feature_data) - WINDOW_SIZE):
#         windows.append(feature_data[i:i+WINDOW_SIZE])

#     X = np.array(windows, dtype=np.float32)

#     # PREDICT: AI Model ko background thread mein bheja taake API block na ho
#     raw_predictions = await asyncio.to_thread(model.predict, X, verbose=0)
#     predictions_scaled = raw_predictions.flatten()

#     # Inverse scale
#     dummy = np.zeros((len(predictions_scaled), len(FEATURES)))
#     dummy[:, 0] = predictions_scaled
#     predictions_actual = scaler.inverse_transform(dummy)[:, 0]

#     # Actual values bhi bhejo
#     actual_values = df["Capacity"].values[WINDOW_SIZE:]

#     return {
#         "battery_id"         : int(df["battery_id"].iloc[0]) if "battery_id" in df.columns else "unknown",
#         "total_cycles"       : len(predictions_actual),
#         "predictions"        : predictions_actual.tolist(),
#         "actual"             : actual_values.tolist(),
#         "mae"                : float(np.mean(np.abs(predictions_actual - actual_values))),
#         "final_soh_percent"  : round(float(predictions_actual[-1] / predictions_actual[0] * 100), 2)
#     }


# @app.post("/analyze")
# async def analyze_with_agent(file: UploadFile = File(...)):

#     # File dobara readable banana (encoding safe)
#     contents = await file.read()
#     try:
#         df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
#     except UnicodeDecodeError:
#         df = pd.read_csv(io.StringIO(contents.decode("latin-1")))

#     missing = [col for col in FEATURES if col not in df.columns]
#     if missing:
#         return {"error": f"Missing columns: {missing}"}

#     if len(df) < WINDOW_SIZE + 1:
#         return {"error": f"Need at least {WINDOW_SIZE + 1} rows"}

#     scaler = MinMaxScaler()
#     df_scaled = df.copy()
#     df_scaled[FEATURES] = scaler.fit_transform(df[FEATURES])

#     feature_data = df_scaled[FEATURES].values
#     windows = []
#     for i in range(len(feature_data) - WINDOW_SIZE):
#         windows.append(feature_data[i:i + WINDOW_SIZE])

#     X = np.array(windows, dtype=np.float32)
    
#     # PREDICT: AI Model ko background thread mein bheja
#     raw_predictions = await asyncio.to_thread(model.predict, X, verbose=0)
#     predictions_scaled = raw_predictions.flatten()

#     dummy = np.zeros((len(predictions_scaled), len(FEATURES)))
#     dummy[:, 0] = predictions_scaled
#     predictions_actual = scaler.inverse_transform(dummy)[:, 0]
#     actual_values = df["Capacity"].values[WINDOW_SIZE:]

#     mae       = float(np.mean(np.abs(predictions_actual - actual_values)))
#     final_soh = round(float(predictions_actual[-1] / predictions_actual[0] * 100), 2)

#     # Agent call karo
#     azure_key = os.getenv("AZURE_OPENAI_API_KEY", "")

#     if azure_key:
#         state = BatteryState(
#             predictions   = predictions_actual.tolist(),
#             actual        = actual_values.tolist(),
#             mae           = mae,
#             final_soh     = final_soh,
#             analysis      = "",
#             recommendation= ""
#         )
        
#         # AGENT INVOKE: Azure OpenAI network call ko background thread mein bheja
#         result = await asyncio.to_thread(battery_agent.invoke, state)
        
#         ai_analysis        = result["analysis"]
#         ai_recommendation  = result["recommendation"]
#     else:
#         ai_analysis       = "Azure OpenAI key not provided. Set AZURE_OPENAI_API_KEY in .env file."
#         ai_recommendation = "Add Azure credentials to .env to enable AI analysis."

#     return {
#         "battery_id"        : int(df["battery_id"].iloc[0]) if "battery_id" in df.columns else "unknown",
#         "total_cycles"      : len(predictions_actual),
#         "predictions"       : predictions_actual.tolist(),
#         "actual"            : actual_values.tolist(),
#         "mae"               : mae,
#         "final_soh_percent" : final_soh,
#         "ai_analysis"       : ai_analysis,
#         "ai_recommendation" : ai_recommendation
#     }

# @app.websocket("/ws/live-stream")
# async def websocket_live_stream(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         await read_stream_and_predict(websocket, model)
#     except WebSocketDisconnect:
#         print("Client disconnected from live stream")





























# main.py — EV Battery SOH Predictor (High-Concurrency Optimized)

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import io
import os

from agent import battery_agent, BatteryState
from fastapi import WebSocket, WebSocketDisconnect
from websocket_consumer import read_stream_and_predict

from dotenv import load_dotenv
load_dotenv()


app = FastAPI(title="EV Battery SOH Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model load karo startup par
tcn_model  = keras.models.load_model("best_tcn_v2.keras")
lstm_model = keras.models.load_model("best_lstm.keras")

def ensemble_predict(X):
    tcn_preds  = tcn_model.predict(X, verbose=0).flatten()
    lstm_preds = lstm_model.predict(X, verbose=0).flatten()
    return (tcn_preds + lstm_preds) / 2.0

FEATURES    = ["Capacity", "Re", "Rct", "ambient_temperature"]
WINDOW_SIZE = 50

@app.get("/")
def root():
    return {"message": "EV Battery SOH Prediction API", "status": "running"}

@app.post("/predict")
async def predict_soh(file: UploadFile = File(...)):
    # CSV read karo (encoding safe)
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except UnicodeDecodeError:
        df = pd.read_csv(io.StringIO(contents.decode("latin-1")))

    # Validate karo
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        return {"error": f"Missing columns: {missing}"}

    if len(df) < WINDOW_SIZE + 1:
        return {"error": f"Need at least {WINDOW_SIZE + 1} rows"}

    # Scale karo
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[FEATURES] = scaler.fit_transform(df[FEATURES])

    # Window banao
    feature_data = df_scaled[FEATURES].values
    windows = []
    for i in range(len(feature_data) - WINDOW_SIZE):
        windows.append(feature_data[i:i+WINDOW_SIZE])

    X = np.array(windows, dtype=np.float32)

    # PREDICT: AI Model ko background thread mein bheja taake API block na ho
    predictions_scaled = await asyncio.to_thread(ensemble_predict, X)

    # Inverse scale
    dummy = np.zeros((len(predictions_scaled), len(FEATURES)))
    dummy[:, 0] = predictions_scaled
    predictions_actual = scaler.inverse_transform(dummy)[:, 0]

    # Actual values bhi bhejo
    actual_values = df["Capacity"].values[WINDOW_SIZE:]

    return {
        "battery_id"         : int(df["battery_id"].iloc[0]) if "battery_id" in df.columns else "unknown",
        "total_cycles"       : len(predictions_actual),
        "predictions"        : predictions_actual.tolist(),
        "actual"             : actual_values.tolist(),
        "mae"                : float(np.mean(np.abs(predictions_actual - actual_values))),
        "final_soh_percent"  : round(float(predictions_actual[-1] / predictions_actual[0] * 100), 2)
    }


@app.post("/analyze")
async def analyze_with_agent(file: UploadFile = File(...)):

    # File dobara readable banana (encoding safe)
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except UnicodeDecodeError:
        df = pd.read_csv(io.StringIO(contents.decode("latin-1")))

    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        return {"error": f"Missing columns: {missing}"}

    if len(df) < WINDOW_SIZE + 1:
        return {"error": f"Need at least {WINDOW_SIZE + 1} rows"}

    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[FEATURES] = scaler.fit_transform(df[FEATURES])

    feature_data = df_scaled[FEATURES].values
    windows = []
    for i in range(len(feature_data) - WINDOW_SIZE):
        windows.append(feature_data[i:i + WINDOW_SIZE])

    X = np.array(windows, dtype=np.float32)
    
    # PREDICT: AI Model ko background thread mein bheja
    predictions_scaled = await asyncio.to_thread(ensemble_predict, X)

    dummy = np.zeros((len(predictions_scaled), len(FEATURES)))
    dummy[:, 0] = predictions_scaled
    predictions_actual = scaler.inverse_transform(dummy)[:, 0]
    actual_values = df["Capacity"].values[WINDOW_SIZE:]

    mae       = float(np.mean(np.abs(predictions_actual - actual_values)))
    final_soh = round(float(predictions_actual[-1] / predictions_actual[0] * 100), 2)

    # Agent call karo
    azure_key = os.getenv("AZURE_OPENAI_API_KEY", "")

    if azure_key:
        state = BatteryState(
            predictions   = predictions_actual.tolist(),
            actual        = actual_values.tolist(),
            mae           = mae,
            final_soh     = final_soh,
            analysis      = "",
            recommendation= ""
        )
        
        # AGENT INVOKE: Azure OpenAI network call ko background thread mein bheja
        result = await asyncio.to_thread(battery_agent.invoke, state)
        
        ai_analysis        = result["analysis"]
        ai_recommendation  = result["recommendation"]
    else:
        ai_analysis       = "Azure OpenAI key not provided. Set AZURE_OPENAI_API_KEY in .env file."
        ai_recommendation = "Add Azure credentials to .env to enable AI analysis."

    return {
        "battery_id"        : int(df["battery_id"].iloc[0]) if "battery_id" in df.columns else "unknown",
        "total_cycles"      : len(predictions_actual),
        "predictions"       : predictions_actual.tolist(),
        "actual"            : actual_values.tolist(),
        "mae"               : mae,
        "final_soh_percent" : final_soh,
        "ai_analysis"       : ai_analysis,
        "ai_recommendation" : ai_recommendation
    }

@app.websocket("/ws/live-stream")
async def websocket_live_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        await read_stream_and_predict(websocket, tcn_model)
    except WebSocketDisconnect:
        print("Client disconnected from live stream")