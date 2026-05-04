"""
generate_architecture_overall.py
Run this in Google Colab or locally to generate docs/architecture_overall.png

Install dependency:
    pip install matplotlib
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 16))
ax.set_xlim(0, 22)
ax.set_ylim(0, 16)
ax.axis('off')
fig.patch.set_facecolor('#0D1117')
ax.set_facecolor('#0D1117')

# ── Color Palette ─────────────────────────────────────────────────────────────
C = {
    'bg':        '#0D1117',
    'box_data':  '#1C3A5E',
    'box_ml':    '#1A3A2A',
    'box_api':   '#3A1C3A',
    'box_agent': '#3A2A0A',
    'box_redis': '#1C1C3A',
    'box_ws':    '#2A1A3A',
    'box_dash':  '#1A2A3A',
    'border_data':'#3B82F6',
    'border_ml': '#22C55E',
    'border_api':'#A855F7',
    'border_agent':'#F59E0B',
    'border_redis':'#6366F1',
    'border_ws': '#EC4899',
    'border_dash':'#06B6D4',
    'text':      '#F0F6FC',
    'subtext':   '#8B949E',
    'arrow':     '#58A6FF',
    'arrow2':    '#3FB950',
}

def box(ax, x, y, w, h, label, sublabel='', fc='#1C3A5E', ec='#3B82F6', fontsize=11, subfontsize=8.5):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=fc, edgecolor=ec, linewidth=2.2,
        zorder=3
    )
    ax.add_patch(rect)
    cy = y + h / 2
    if sublabel:
        ax.text(x + w/2, cy + 0.18, label,
                ha='center', va='center', fontsize=fontsize,
                color=C['text'], fontweight='bold', zorder=4)
        ax.text(x + w/2, cy - 0.28, sublabel,
                ha='center', va='center', fontsize=subfontsize,
                color=C['subtext'], zorder=4, style='italic')
    else:
        ax.text(x + w/2, cy, label,
                ha='center', va='center', fontsize=fontsize,
                color=C['text'], fontweight='bold', zorder=4)

def arrow(ax, x1, y1, x2, y2, label='', color='#58A6FF', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->', color=color,
                    lw=lw, connectionstyle='arc3,rad=0.0'
                ), zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.12, my, label,
                ha='left', va='center', fontsize=7.5,
                color=color, zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc=C['bg'], ec='none', alpha=0.8))

def section_label(ax, x, y, text, color):
    ax.text(x, y, text, ha='left', va='center', fontsize=8,
            color=color, fontweight='bold', alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.2', fc=C['bg'], ec=color, linewidth=0.8, alpha=0.5))

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(11, 15.4, 'EV Battery SOH Prediction — Overall System Architecture',
        ha='center', va='center', fontsize=16, color=C['text'],
        fontweight='bold')
ax.text(11, 14.95, 'Data Flow: Ingestion  →  ML Inference  →  Agentic Reasoning  →  Real-Time Observability',
        ha='center', va='center', fontsize=9.5, color=C['subtext'])

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — DATA SOURCES (y=13.0)
# ══════════════════════════════════════════════════════════════════════════════
section_label(ax, 0.3, 13.85, '[ DATA SOURCES ]', C['border_data'])

box(ax, 0.4,  12.8, 3.2, 0.95, 'CSV / IoT Telemetry',
    'Battery_Data_Cleaned.csv', C['box_data'], C['border_data'])

box(ax, 4.2,  12.8, 3.2, 0.95, 'HTTP Clients',
    'Analysts / Applications', C['box_data'], C['border_data'])

box(ax, 8.0,  12.8, 3.2, 0.95, 'Fleet IoT Devices',
    'EVFleetSystemUser (Locust)', C['box_data'], C['border_data'])

box(ax, 11.8, 12.8, 3.2, 0.95, 'Redis Stream Simulator',
    'stream_simulator.py', C['box_redis'], C['border_redis'])

box(ax, 15.6, 12.8, 3.2, 0.95, 'Kafka Producer',
    'kafka_streamer.py  ~20 msg/s', C['box_redis'], C['border_redis'])

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — API GATEWAY + MESSAGE LAYER (y=11.0)
# ══════════════════════════════════════════════════════════════════════════════
section_label(ax, 0.3, 12.15, '[ API GATEWAY & MESSAGE LAYER ]', C['border_api'])

box(ax, 3.0, 11.1, 6.0, 0.95, 'FastAPI Backend  (Uvicorn)',
    'main.py  |  port 8000  |  asyncio event loop', C['box_api'], C['border_api'], fontsize=12)

box(ax, 11.8, 11.1, 3.2, 0.95, 'Redis Streams',
    'XADD ev_battery_stream', C['box_redis'], C['border_redis'])

box(ax, 15.6, 11.1, 3.2, 0.95, 'Kafka Topic',
    'ev_battery_telemetry  |  3 partitions', C['box_redis'], C['border_redis'])

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — ENDPOINTS + CONSUMERS (y=9.2)
# ══════════════════════════════════════════════════════════════════════════════
section_label(ax, 0.3, 10.45, '[ ENDPOINTS & CONSUMERS ]', C['border_ml'])

box(ax, 0.4,  9.3, 2.6, 0.95, 'GET /',
    'Health Check', C['box_ml'], C['border_ml'], fontsize=10)

box(ax, 3.2,  9.3, 2.6, 0.95, 'POST /predict',
    'CSV → TCN → SOH%', C['box_ml'], C['border_ml'], fontsize=10)

box(ax, 6.0,  9.3, 2.8, 0.95, 'POST /analyze',
    'TCN + LangGraph Agent', C['box_agent'], C['border_agent'], fontsize=10)

box(ax, 9.0,  9.3, 2.6, 0.95, 'WS /ws/live-stream',
    'Redis → WebSocket', C['box_ws'], C['border_ws'], fontsize=9.5)

box(ax, 11.8, 9.3, 3.2, 0.95, 'WebSocket Consumer',
    'websocket_consumer.py', C['box_ws'], C['border_ws'])

box(ax, 15.6, 9.3, 3.2, 0.95, 'Kafka Consumer',
    'kafka_consumer_direct.py', C['box_redis'], C['border_redis'])

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — ML INFERENCE (y=7.4)
# ══════════════════════════════════════════════════════════════════════════════
section_label(ax, 0.3, 8.65, '[ ML INFERENCE LAYER ]', C['border_ml'])

box(ax, 1.8, 7.5, 8.2, 0.95,
    'TCN Model  (best_tcn_v2.keras)',
    'Input: (50, 4)  |  Causal Conv1D  |  Dilation 1,2,4,8  |  MAE target < 0.12 Ah',
    C['box_ml'], C['border_ml'], fontsize=11)

box(ax, 11.8, 7.5, 3.2, 0.95, 'Sliding Window Buffer',
    '50 cycles per battery_id', C['box_ml'], C['border_ml'])

box(ax, 15.6, 7.5, 3.2, 0.95, 'Per-Battery Deque',
    'max 200 cycles | deque()', C['box_ml'], C['border_ml'])

# ══════════════════════════════════════════════════════════════════════════════
# ROW 5 — AGENTIC LAYER (y=5.6)
# ══════════════════════════════════════════════════════════════════════════════
section_label(ax, 0.3, 6.85, '[ AGENTIC AI LAYER ]', C['border_agent'])

box(ax, 0.4, 5.7, 3.8, 0.95, 'LangGraph Node 1',
    'analyze_degradation()', C['box_agent'], C['border_agent'])

box(ax, 4.4, 5.7, 3.8, 0.95, 'LangGraph Node 2',
    'give_recommendation()', C['box_agent'], C['border_agent'])

box(ax, 8.4, 5.7, 3.8, 0.95, 'Azure OpenAI (GPT-4)',
    'Natural language report', C['box_agent'], C['border_agent'])

# SOH Decision box
box(ax, 12.4, 5.3, 5.4, 1.35, 'SOH Classification',
    '>= 85%  HEALTHY  |  70-84%  CAUTION  |  <70%  REPLACE',
    C['box_ml'], C['border_ml'], fontsize=10)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 6 — OUTPUT LAYER (y=3.8)
# ══════════════════════════════════════════════════════════════════════════════
section_label(ax, 0.3, 5.05, '[ OUTPUT & OBSERVABILITY ]', C['border_dash'])

box(ax, 0.4,  3.9, 3.8, 0.95, 'JSON API Response',
    'predictions + MAE + SOH%', C['box_dash'], C['border_dash'])

box(ax, 4.4,  3.9, 3.8, 0.95, 'NL Maintenance Report',
    'LangGraph analysis output', C['box_dash'], C['border_dash'])

box(ax, 8.4,  3.9, 3.8, 0.95, 'Live WebSocket Push',
    'SOH payload → frontend', C['box_ws'], C['border_ws'])

box(ax, 12.4, 3.9, 4.0, 0.95, 'Chart.js Dashboard',
    'index.html | port 3000', C['box_dash'], C['border_dash'])

box(ax, 16.6, 3.9, 3.8, 0.95, 'Console Predictions',
    'Battery | Cycle | SOH | Status', C['box_ml'], C['border_ml'])

# ══════════════════════════════════════════════════════════════════════════════
# ROW 7 — LOAD TESTING (y=2.1)
# ══════════════════════════════════════════════════════════════════════════════
section_label(ax, 0.3, 3.2, '[ LOAD TESTING ]', C['border_api'])

box(ax, 1.0, 2.2, 4.5, 0.85, 'Locust  —  BatteryAPIUser',
    '/predict (w3)  /analyze (w1)  / (w1)  |  1-3s delays',
    C['box_api'], C['border_api'], fontsize=9)

box(ax, 6.0, 2.2, 4.5, 0.85, 'Locust  —  EVFleetSystemUser',
    '/predict fleet  |  0.1-0.5s delays  |  IoT simulation',
    C['box_api'], C['border_api'], fontsize=9)

box(ax, 11.0, 2.2, 5.0, 0.85, 'Result: 0% failures on /predict',
    '260 concurrent users  |  Failures: Azure API timeout only',
    C['box_ml'], C['border_ml'], fontsize=9)

box(ax, 16.5, 2.2, 5.1, 0.85, 'Scale Path: asyncio.to_thread',
    'uvicorn --workers N  |  containerized horizontal scale',
    C['box_dash'], C['border_dash'], fontsize=9)

# ══════════════════════════════════════════════════════════════════════════════
# ARROWS — ROW 1 → ROW 2
# ══════════════════════════════════════════════════════════════════════════════
# CSV → FastAPI
arrow(ax, 2.0, 12.8, 5.0, 12.05, 'HTTP POST /predict', C['arrow'])
# HTTP Clients → FastAPI
arrow(ax, 5.8, 12.8, 6.0, 12.05, '', C['arrow'])
# Fleet → FastAPI
arrow(ax, 9.6, 12.8, 8.0, 12.05, '', C['arrow'])
# Redis Simulator → Redis Streams
arrow(ax, 13.4, 12.8, 13.4, 12.05, 'XADD', C['border_redis'])
# Kafka Producer → Kafka Topic
arrow(ax, 17.2, 12.8, 17.2, 12.05, 'publish', C['border_redis'])

# ══════════════════════════════════════════════════════════════════════════════
# ARROWS — ROW 2 → ROW 3
# ══════════════════════════════════════════════════════════════════════════════
# FastAPI → endpoints (fan out)
arrow(ax, 4.5, 11.1, 1.7, 10.25, '', C['arrow'])
arrow(ax, 5.5, 11.1, 4.5, 10.25, '', C['arrow'])
arrow(ax, 6.5, 11.1, 7.4, 10.25, '', C['agent_color'] if False else C['border_agent'])
arrow(ax, 7.5, 11.1, 10.3, 10.25, '', C['border_ws'])
# Redis Streams → WebSocket Consumer
arrow(ax, 13.4, 11.1, 13.4, 10.25, 'XREAD', C['border_redis'])
# Kafka Topic → Kafka Consumer
arrow(ax, 17.2, 11.1, 17.2, 10.25, 'consume', C['border_redis'])

# ══════════════════════════════════════════════════════════════════════════════
# ARROWS — ROW 3 → ROW 4
# ══════════════════════════════════════════════════════════════════════════════
# /predict → TCN
arrow(ax, 4.5, 9.3, 5.0, 8.45, '', C['arrow2'])
# /analyze → TCN
arrow(ax, 7.4, 9.3, 6.5, 8.45, '', C['border_agent'])
# WebSocket Consumer → Sliding Window
arrow(ax, 13.4, 9.3, 13.4, 8.45, '50-cycle buffer', C['border_redis'])
# Kafka Consumer → Per-Battery Deque
arrow(ax, 17.2, 9.3, 17.2, 8.45, 'accumulate', C['border_redis'])

# ══════════════════════════════════════════════════════════════════════════════
# ARROWS — ROW 4 → ROW 5
# ══════════════════════════════════════════════════════════════════════════════
# TCN → LangGraph Node 1
arrow(ax, 5.0, 7.5, 2.3, 6.65, '', C['border_agent'])
# TCN → JSON output (predict)
arrow(ax, 5.9, 7.5, 2.3, 4.85, '', C['arrow'])
# Sliding Window → TCN (merge back)
arrow(ax, 13.4, 7.5, 9.8, 8.45, 'inference', C['arrow2'])
# Deque → TCN merge
arrow(ax, 17.2, 7.5, 9.8, 8.2, '', C['arrow2'])
# TCN → SOH Classification
arrow(ax, 9.8, 7.5, 13.4, 6.65, 'SOH%', C['border_ml'])

# ══════════════════════════════════════════════════════════════════════════════
# ARROWS — ROW 5 → ROW 6
# ══════════════════════════════════════════════════════════════════════════════
# Node1 → Node2
arrow(ax, 4.2, 6.17, 4.4, 6.17, '', C['border_agent'])
# Node2 → Azure
arrow(ax, 8.2, 6.17, 8.4, 6.17, '', C['border_agent'])
# Azure → NL Report
arrow(ax, 10.3, 5.7, 6.3, 4.85, 'report', C['border_agent'])
# SOH Classification → Dashboard
arrow(ax, 15.1, 5.3, 14.4, 4.85, '', C['border_dash'])
# SOH Classification → Console
arrow(ax, 17.6, 5.3, 18.5, 4.85, '', C['border_ml'])

# WebSocket Consumer → Live Push
arrow(ax, 13.4, 9.3, 10.3, 4.85, 'WS push', C['border_ws'])

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (C['border_data'],  'Data Sources'),
    (C['border_ml'],    'ML / Inference'),
    (C['border_agent'], 'Agentic AI'),
    (C['border_api'],   'API / Load Test'),
    (C['border_redis'], 'Streaming (Redis/Kafka)'),
    (C['border_ws'],    'WebSocket'),
    (C['border_dash'],  'Dashboard / Output'),
]
for i, (color, label) in enumerate(legend_items):
    rx = 0.5 + i * 3.05
    rect = FancyBboxPatch((rx, 0.5), 2.7, 0.55,
                           boxstyle="round,pad=0.05",
                           facecolor=C['bg'], edgecolor=color, linewidth=1.8)
    ax.add_patch(rect)
    ax.text(rx + 1.35, 0.775, label,
            ha='center', va='center', fontsize=7.5,
            color=color, fontweight='bold')

plt.tight_layout(pad=0.3)
plt.savefig('architecture_overall.png', dpi=180, bbox_inches='tight',
            facecolor=C['bg'])
print("Saved: architecture_overall.png")
plt.close()