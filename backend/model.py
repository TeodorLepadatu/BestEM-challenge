import sys
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

# 1. Set backend to 'Agg' to prevent GUI errors
plt.switch_backend('Agg')

def predict_disease(file_path, disease_name):
    # --- GPU CONFIGURATION ---
    # This automatically uses the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    LOOK_BACK = 15
    BATCH_SIZE = 16
    EPOCHS = 100 
    LEARNING_RATE = 0.001

    # --- Data Processing ---
    records = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    patient = json.loads(line)
                    
                    # Normalize and find diagnosis
                    diagnosis = patient.get('diagnosis', '')
                    if not diagnosis and 'final_diagnostic' in patient:
                         diagnosis = patient['final_diagnostic'].get('most_probable_diagnostic', '')
                    
                    # Case-insensitive check
                    if disease_name.lower() in str(diagnosis).lower():
                        date_str = patient.get('date') or patient.get('report_date')
                        if date_str:
                            records.append({'date': date_str})
                    
                    # REMOVED print("processed data") - This breaks Node.js
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return None, None, "Data file not found"

    if not records:
        return 0, None, None # No data found

    df_raw = pd.DataFrame(records)
    df_raw['date'] = pd.to_datetime(df_raw['date'], utc=True)
    df_raw = df_raw.sort_values('date')

    # Aggregate to Weekly counts
    df_weekly = df_raw.groupby(pd.Grouper(key='date', freq='W-MON')).size().reset_index(name='cases')
    df_weekly.set_index('date', inplace=True)
    
    # Fill missing weeks with 0
    if not df_weekly.empty:
        full_idx = pd.date_range(start=df_weekly.index.min(), end=df_weekly.index.max(), freq='W-MON')
        df_weekly = df_weekly.reindex(full_idx, fill_value=0)
    
    # Smoothing
    df_weekly['cases'] = df_weekly['cases'].rolling(window=4, min_periods=1).mean()
    
    # Prepare for LSTM
    raw_values = df_weekly['cases'].values.reshape(-1, 1)
    
    # Not enough data fallback
    if len(raw_values) < LOOK_BACK + 2:
         return int(raw_values[-1][0]) if len(raw_values) > 0 else 0, None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(raw_values)

    def create_sequences(data, seq_length):
        xs, ys = [], []
        if len(data) <= seq_length:
            return np.array([]), np.array([])
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    X, y = create_sequences(scaled_data, LOOK_BACK)
    
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()

    # --- LSTM Model ---
    class InfluenzaLSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, output_size=1):
            super(InfluenzaLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
            self.linear = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            last_step = out[:, -1, :]
            prediction = self.linear(last_step)
            return prediction

    model = InfluenzaLSTM().to(device) # Move model to GPU
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # REMOVED print("started training") - This breaks Node.js

    # --- Training ---
    train_dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        for seq, label in train_loader:
            # Move data to GPU
            seq, label = seq.to(device), label.to(device)
            
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

    # --- Prediction ---
    model.eval()
    last_sequence = scaled_data[-LOOK_BACK:]
    # Move input to GPU
    input_tensor = torch.tensor(last_sequence).float().unsqueeze(0).to(device)

    with torch.no_grad():
        next_week_scaled = model(input_tensor)
        next_week_cases = scaler.inverse_transform(next_week_scaled.cpu().numpy())
    
    prediction_result = max(0, int(next_week_cases[0][0]))

    # --- Generate Plot ---
    img_base64 = "" # FIX: Removed random text
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(df_weekly.index, df_weekly['cases'], label='Historical (Smoothed)', color='#1f77b4', linewidth=2)
        
        last_date = df_weekly.index[-1]
        next_date = last_date + pd.Timedelta(weeks=1)
        
        # FIX: Removed random text at end of line
        plt.scatter([next_date], [prediction_result], color='#ff7f0e', s=100, label='AI Prediction', zorder=5)
        
        plt.title(f"Forecast: {disease_name}")
        plt.xlabel("Date")
        plt.ylabel("Weekly Cases")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    except Exception:
        pass 

    return prediction_result, img_base64, None

# --- Main Entry Point ---
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Missing arguments"}))
        sys.exit(1)

    file_path = sys.argv[1]
    disease_name = sys.argv[2]

    pred, img, err = predict_disease(file_path, disease_name)

    if err:
        print(json.dumps({"error": err}))
    else:
        # This is the ONLY thing that should be printed
        print(json.dumps({
            "prediction": pred,
            "image_base64": img
        }))