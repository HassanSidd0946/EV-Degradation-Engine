# stream_simulator.py
# Ye script Battery data ko live sensor ki tarah stream karta hai

import redis
import pandas as pd
import json
import time
import sys

# Redis se connect karo
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def stream_battery_data(csv_path: str, battery_id: int = 54, delay: float = 1.0):
    
    df = pd.read_csv(csv_path)
    df = df[df['battery_id'] == battery_id].copy()
    df = df[['battery_id', 'test_id', 'Capacity', 
             'Re', 'Rct', 'ambient_temperature']].reset_index(drop=True)

    print(f"Starting stream for Battery {battery_id}")
    print(f"Total cycles to stream: {len(df)}")
    print(f"Delay between cycles: {delay}s")
    print("-" * 40)

    # Redis stream clear karo pehle
    try:
        r.delete('battery:stream')
    except:
        pass

    for idx, row in df.iterrows():
        data = {
            'battery_id'         : str(int(row['battery_id'])),
            'cycle'              : str(int(row['test_id'])),
            'capacity'           : str(round(float(row['Capacity']), 6)),
            're'                 : str(round(float(row['Re']), 6)),
            'rct'                : str(round(float(row['Rct']), 6)),
            'ambient_temperature': str(round(float(row['ambient_temperature']), 2)),
            'timestamp'          : str(time.time())
        }

        # Redis stream mein publish karo
        r.xadd('battery:stream', data)

        print(f"Cycle {int(row['test_id']):>4}  |  "
              f"Capacity: {row['Capacity']:.4f} Ah  |  "
              f"Temp: {row['ambient_temperature']}°C")

        time.sleep(delay)

    print("-" * 40)
    print("Stream complete.")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "Battery_Data_Cleaned.csv"
    battery  = int(sys.argv[2]) if len(sys.argv) > 2 else 54
    delay    = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    stream_battery_data(csv_file, battery, delay)