"""
generate_architecture_kafka.py
Run this in Google Colab or locally to generate docs/architecture_kafka.png

Install dependency:
    pip install matplotlib
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 14))
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis('off')
fig.patch.set_facecolor('#0D1117')
ax.set_facecolor('#0D1117')

C = {
    'bg':         '#0D1117',
    'kafka':      '#1A2A0A',
    'kafka_b':    '#22C55E',
    'zk':         '#0A1A2A',
    'zk_b':       '#3B82F6',
    'producer':   '#2A1A0A',
    'producer_b': '#F59E0B',
    'consumer':   '#1A0A2A',
    'consumer_b': '#A855F7',
    'spark':      '#0A2A2A',
    'spark_b':    '#06B6D4',
    'redis':      '#1C1C3A',
    'redis_b':    '#6366F1',
    'tcn':        '#1A3A2A',
    'tcn_b':      '#4ADE80',
    'output':     '#1A1A3A',
    'output_b':   '#818CF8',
    'scale':      '#2A0A1A',
    'scale_b':    '#F472B6',
    'text':       '#F0F6FC',
    'subtext':    '#8B949E',
    'part0':      '#1E3A1E',
    'part1':      '#1E2A3A',
    'part2':      '#3A1E2A',
}

def box(ax, x, y, w, h, label, sublabel='', fc='#1C3A5E', ec='#3B82F6',
        fontsize=10.5, subfontsize=8, bold=True):
    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.09",
                           facecolor=fc, edgecolor=ec, linewidth=2.2, zorder=3)
    ax.add_patch(rect)
    cy = y + h / 2
    fw = 'bold' if bold else 'normal'
    if sublabel:
        ax.text(x+w/2, cy+0.2,  label,    ha='center', va='center',
                fontsize=fontsize, color=C['text'], fontweight=fw, zorder=4)
        ax.text(x+w/2, cy-0.25, sublabel, ha='center', va='center',
                fontsize=subfontsize, color=C['subtext'], style='italic', zorder=4)
    else:
        ax.text(x+w/2, cy, label, ha='center', va='center',
                fontsize=fontsize, color=C['text'], fontweight=fw, zorder=4)

def arrow(ax, x1, y1, x2, y2, label='', color='#58A6FF', lw=2.0, rad=0.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->', color=color, lw=lw,
                    connectionstyle=f'arc3,rad={rad}'
                ), zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.1, my+0.1, label, ha='left', va='bottom',
                fontsize=7.5, color=color, zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc=C['bg'], ec='none', alpha=0.85))

def section_bg(ax, x, y, w, h, color, label):
    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.15",
                           facecolor=color+'18', edgecolor=color,
                           linewidth=1.2, linestyle='--', zorder=1, alpha=0.6)
    ax.add_patch(rect)
    ax.text(x+0.25, y+h-0.25, label, ha='left', va='top',
            fontsize=8, color=color, fontweight='bold', alpha=0.8, zorder=2)

def section_label(ax, x, y, text, color):
    ax.text(x, y, text, ha='left', va='center', fontsize=8,
            color=color, fontweight='bold', alpha=0.75,
            bbox=dict(boxstyle='round,pad=0.2', fc=C['bg'], ec=color,
                      linewidth=0.8, alpha=0.5))

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(11, 13.5, 'EV Battery SOH — Kafka Scalability & Streaming Architecture',
        ha='center', va='center', fontsize=16, color=C['text'], fontweight='bold')
ax.text(11, 13.05,
        'Fleet-scale IoT telemetry ingestion  →  Partitioned message queue  →  Per-battery stateful inference  →  Scalable output sinks',
        ha='center', va='center', fontsize=9, color=C['subtext'])

# ══════════════════════════════════════════════════════════════════════════════
# SECTION BACKGROUNDS
# ══════════════════════════════════════════════════════════════════════════════
section_bg(ax, 0.3,  11.2, 4.5,  1.35, C['producer_b'], 'PRODUCER LAYER')
section_bg(ax, 5.2,  9.5,  11.6, 3.1,  C['kafka_b'],    'KAFKA CLUSTER')
section_bg(ax, 5.2,  7.2,  11.6, 2.0,  C['consumer_b'], 'CONSUMER GROUP')
section_bg(ax, 0.3,  5.0,  10.8, 1.9,  C['spark_b'],    'PYSPARK STRUCTURED STREAMING (Linux / Cloud)')
section_bg(ax, 0.3,  2.8,  10.8, 1.9,  C['tcn_b'],      'TCN INFERENCE ENGINE')
section_bg(ax, 11.5, 7.2,  10.2, 5.5,  C['redis_b'],    'REDIS STREAMS PIPELINE')
section_bg(ax, 0.3,  0.5,  21.4, 2.0,  C['output_b'],   'OUTPUT & SCALE PATH')

# ══════════════════════════════════════════════════════════════════════════════
# PRODUCER LAYER
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 0.5, 11.5, 4.0, 0.85,
    'kafka_streamer.py',
    'CSV infinite loop  |  battery_id → partition key  |  ~20 msg/s',
    C['producer'], C['producer_b'], fontsize=10)

# Throughput annotation
ax.text(2.5, 11.42, '~1.7 M messages / day', ha='center', va='top',
        fontsize=7.5, color=C['producer_b'], style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# ZOOKEEPER
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 0.5, 9.8, 4.0, 0.75, 'ZooKeeper',
    'port 2181  |  broker coordination', C['zk'], C['zk_b'], fontsize=10)

arrow(ax, 4.5, 10.17, 5.5, 10.17, 'registers', C['zk_b'], lw=1.5)

# ══════════════════════════════════════════════════════════════════════════════
# KAFKA BROKER + TOPIC
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 5.5, 11.5, 4.5, 0.85, 'Kafka Broker  (id=0)',
    'port 9092  |  Kafka 3.7.0  |  Java 11',
    C['kafka'], C['kafka_b'], fontsize=10)

# Topic label
ax.text(10.8, 11.92, 'Topic: ev_battery_telemetry',
        ha='left', va='center', fontsize=9, color=C['kafka_b'], fontweight='bold')

# Three partitions
part_colors = [C['part0'], C['part1'], C['part2']]
part_borders = ['#22C55E', '#3B82F6', '#EC4899']
part_labels  = ['Partition 0', 'Partition 1', 'Partition 2']
for i in range(3):
    px = 5.5 + i * 3.7
    box(ax, px, 9.7, 3.3, 1.55,
        part_labels[i],
        f'battery_id % 3 == {i}\noffset-based ordering',
        part_colors[i], part_borders[i], fontsize=9.5, subfontsize=7.5)

# Producer → Broker
arrow(ax, 4.5, 11.92, 5.5, 11.92, 'produce\n(battery_id key)', C['producer_b'], lw=2.2)

# Broker → Partitions (fan out)
for i in range(3):
    px_center = 5.5 + i * 3.7 + 1.65
    arrow(ax, 7.75, 11.5, px_center, 11.25, '', C['kafka_b'], lw=1.8)

# ══════════════════════════════════════════════════════════════════════════════
# CONSUMER GROUP
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 5.5,  7.5, 3.3, 0.85, 'Consumer Instance 0',
    'kafka_consumer_direct.py', C['consumer'], C['consumer_b'], fontsize=9.5)
box(ax, 9.2,  7.5, 3.3, 0.85, 'Consumer Instance 1',
    '(horizontal scale)', C['consumer'], C['consumer_b'], fontsize=9.5)
box(ax, 12.9, 7.5, 3.3, 0.85, 'Consumer Instance 2',
    '(horizontal scale)', C['consumer'], C['consumer_b'], fontsize=9.5)

# Group label
ax.text(10.35, 7.35, 'Consumer Group: ev_soh_consumer_group',
        ha='center', va='top', fontsize=8, color=C['consumer_b'], style='italic')

# Partitions → Consumers
for i in range(3):
    px_top    = 5.5 + i * 3.7 + 1.65
    px_bottom = 5.5 + i * 3.7 + 1.65
    arrow(ax, px_top, 9.7, px_bottom, 8.35, '', C['kafka_b'], lw=1.8)

# ══════════════════════════════════════════════════════════════════════════════
# PYSPARK STRUCTURED STREAMING (Linux / Cloud)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 0.5,  5.25, 3.8, 0.95, 'Spark readStream',
    '.format("kafka")\n.option("subscribe", topic)',
    C['spark'], C['spark_b'], fontsize=9, subfontsize=7)

box(ax, 4.6,  5.25, 3.5, 0.95, 'foreachBatch()',
    'stateful per-battery\nbuffer accumulation',
    C['spark'], C['spark_b'], fontsize=9, subfontsize=7)

box(ax, 8.3,  5.25, 2.8, 0.95, 'Trigger',
    'processingTime\n= 5 seconds',
    C['spark'], C['spark_b'], fontsize=9, subfontsize=7)

arrow(ax, 4.3, 5.72, 4.6, 5.72, '', C['spark_b'], lw=1.8)
arrow(ax, 8.1, 5.72, 8.3, 5.72, '', C['spark_b'], lw=1.8)

# Consumer → PySpark
arrow(ax, 5.5, 7.5, 2.4, 6.2, 'Kafka source', C['spark_b'], lw=1.5, rad=-0.2)

# PySpark note
ax.text(0.5, 5.15, 'Note: pyspark_consumer.py — production-ready for Linux/cloud. winutils.exe constraint on Windows resolved by kafka_consumer_direct.py.',
        ha='left', va='top', fontsize=7, color=C['subtext'], style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# TCN INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 0.5, 3.05, 3.8, 0.95, 'Per-Battery Deque',
    'max 200 cycles\ndeque(maxlen=200)',
    C['tcn'], C['tcn_b'], fontsize=9.5, subfontsize=7.5)

box(ax, 4.6, 3.05, 3.5, 0.95, 'Sliding Window',
    'last 50 cycles → (50,4)\nMinMaxScaler per window',
    C['tcn'], C['tcn_b'], fontsize=9.5, subfontsize=7.5)

box(ax, 8.3, 3.05, 2.8, 0.95, 'TCN Inference',
    'best_tcn_v2.keras\nMAE < 0.12 Ah',
    C['tcn'], C['tcn_b'], fontsize=9.5, subfontsize=7.5)

arrow(ax, 4.3, 3.52, 4.6, 3.52, '', C['tcn_b'], lw=1.8)
arrow(ax, 8.1, 3.52, 8.3, 3.52, '', C['tcn_b'], lw=1.8)

# Consumer → Deque
arrow(ax, 6.15, 7.5, 2.4, 4.0, 'accumulate', C['tcn_b'], lw=1.5, rad=0.15)

# PySpark → Deque
arrow(ax, 2.4, 5.25, 2.4, 4.0, '', C['tcn_b'], lw=1.5)

# ══════════════════════════════════════════════════════════════════════════════
# REDIS STREAMS PIPELINE (right side)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 11.8, 11.5, 4.2, 0.85, 'stream_simulator.py',
    'CSV row-by-row  |  configurable delay',
    C['redis'], C['redis_b'], fontsize=9.5)

box(ax, 16.3, 11.5, 4.8, 0.85, 'Redis Streams',
    'XADD ev_battery_stream  |  port 6379',
    C['redis'], C['redis_b'], fontsize=9.5)

box(ax, 11.8, 9.8, 4.2, 0.85, 'websocket_consumer.py',
    'XREAD blocking  |  50-cycle buffer',
    C['redis'], C['redis_b'], fontsize=9.5)

box(ax, 16.3, 9.8, 4.8, 0.85, 'TCN Inference',
    '(50,4) window  |  SOH% computed',
    C['tcn'], C['tcn_b'], fontsize=9.5)

box(ax, 11.8, 8.1, 4.2, 0.85, 'FastAPI WebSocket',
    '/ws/live-stream  |  JSON push',
    C['redis'], C['redis_b'], fontsize=9.5)

box(ax, 16.3, 8.1, 4.8, 0.85, 'Chart.js Dashboard',
    'index.html  |  port 3000  |  live curve',
    C['output'], C['output_b'], fontsize=9.5)

# Redis pipeline arrows
arrow(ax, 16.0, 11.92, 16.3, 11.92, 'XADD', C['redis_b'], lw=1.8)
arrow(ax, 13.9, 11.5, 13.9, 10.65, 'XREAD', C['redis_b'], lw=1.8)
arrow(ax, 16.3, 10.65, 21.1, 10.65, '', C['tcn_b'], lw=1.5, rad=0.2)
# Fix: redis consumer → websocket
arrow(ax, 13.9, 9.8, 13.9, 8.95, 'push prediction', C['redis_b'], lw=1.8)
arrow(ax, 18.7, 9.8, 18.7, 8.95, '', C['tcn_b'], lw=1.5)
arrow(ax, 16.0, 8.52, 16.3, 8.52, '', C['output_b'], lw=1.8)

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT & SCALE PATH (bottom)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 0.5, 0.75, 3.5, 0.85, 'Console Output',
    'Battery | Cycle | Actual | Predicted | SOH | Status',
    C['output'], C['output_b'], fontsize=8.5, subfontsize=7)

box(ax, 4.2, 0.75, 3.5, 0.85, 'API Response (JSON)',
    '/predict  →  SOH%  +  MAE  +  predictions[]',
    C['output'], C['output_b'], fontsize=8.5, subfontsize=7)

box(ax, 7.9, 0.75, 3.5, 0.85, 'NL Report (LangGraph)',
    '/analyze  →  Azure OpenAI  →  maintenance action',
    C['output'], C['output_b'], fontsize=8.5, subfontsize=7)

box(ax, 11.8, 0.75, 4.2, 0.85, 'Horizontal Scale Path',
    'Consumer Group  |  uvicorn --workers N  |  Docker containers',
    C['scale'], C['scale_b'], fontsize=8.5, subfontsize=7)

box(ax, 16.3, 0.75, 5.0, 0.85, 'Cloud Deployment Target',
    'GCP Pub/Sub  |  AWS Kinesis  |  Azure Event Hubs',
    C['scale'], C['scale_b'], fontsize=8.5, subfontsize=7)

# TCN → Console
arrow(ax, 9.65, 3.05, 2.25, 1.6, 'prediction', C['output_b'], lw=1.5, rad=0.1)

# Dashboard → (already connected above)

# Scale arrows
arrow(ax, 9.65, 10.17, 11.8, 0.75+0.425, '', C['scale_b'], lw=1.2, rad=0.3)

# ══════════════════════════════════════════════════════════════════════════════
# THROUGHPUT ANNOTATIONS (floating stats)
# ══════════════════════════════════════════════════════════════════════════════
stats = [
    (7.75, 9.3,  '787 msgs\n112 sec run',  C['kafka_b']),
    (7.75, 8.85, '688 predictions\n7 msg/s', C['tcn_b']),
]
for sx, sy, st, sc in stats:
    ax.text(sx, sy, st, ha='center', va='center', fontsize=7,
            color=sc, style='italic', zorder=6,
            bbox=dict(boxstyle='round,pad=0.2', fc=C['bg'], ec=sc,
                      linewidth=0.8, alpha=0.7))

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (C['producer_b'], 'Producer'),
    (C['kafka_b'],    'Kafka Cluster'),
    (C['consumer_b'], 'Consumer Group'),
    (C['spark_b'],    'PySpark (Linux)'),
    (C['tcn_b'],      'TCN Inference'),
    (C['redis_b'],    'Redis Streams'),
    (C['output_b'],   'Output Layer'),
    (C['scale_b'],    'Scale / Cloud'),
]
legend_w = 22 / len(legend_items)
for i, (color, label) in enumerate(legend_items):
    rx = i * legend_w + 0.1
    rect = FancyBboxPatch((rx, 0.08), legend_w - 0.2, 0.42,
                           boxstyle="round,pad=0.05",
                           facecolor=C['bg'], edgecolor=color, linewidth=1.8)
    ax.add_patch(rect)
    ax.text(rx + (legend_w-0.2)/2, 0.29, label,
            ha='center', va='center', fontsize=7.8,
            color=color, fontweight='bold')

plt.tight_layout(pad=0.3)
plt.savefig('architecture_kafka.png', dpi=180, bbox_inches='tight',
            facecolor=C['bg'])
print("Saved: architecture_kafka.png")
plt.close()