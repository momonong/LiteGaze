"""
LexiGaze v8 推理速度視覺化
繪製三張子圖：
  1. process_file v7 vs v8 速度比較（Bar chart）
  2. 各文本長度 per-word 延遲（Bar chart）
  3. 單句 run() 各階段耗時拆解（Stacked bar）
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.rcParams['font.family'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 色彩 ──────────────────────────────────────────────────────────
C_V7   = "#E05C5C"
C_V8   = "#6ABF69"
C_SPACY = "#4C9BE8"
C_GPT2  = "#F4A44A"
C_SCORE = "#B97EE8"
C_BG   = "#1E1E2E"
C_AX   = "#2A2A3E"
C_SP   = "#555577"
C_TEXT = "white"

def style_ax(ax):
    ax.set_facecolor(C_AX)
    ax.tick_params(colors=C_TEXT)
    ax.xaxis.label.set_color(C_TEXT)
    ax.yaxis.label.set_color(C_TEXT)
    ax.title.set_color(C_TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(C_SP)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='white')
    ax.set_axisbelow(True)

# ── 測量數據（來自 bench_speed.py 實測結果） ─────────────────────
# 1. process_file: v7 vs v8（25 句，330 詞）
pf_labels  = ["v7\n(逐句 run)", "v8\n(chunked)"]
pf_times   = [833.8, 289.1]   # ms

# 2. 不同文本長度 per-word 延遲
len_labels = ["Short\n(1句, 11詞)", "Medium\n(5句, 65詞)", "Long\n(25句, 330詞)"]
len_ms     = [32.4, 74.4, 337.7]   # total ms
len_words  = [11, 65, 330]
per_word   = [ms / w for ms, w in zip(len_ms, len_words)]

# 3. 各階段拆解（Medium 文本）
stages     = ["spaCy\n解析", "GPT-2\n推理", "Scoring\n+其他"]
stage_ms   = [14.8, 63.0, 2.7]     # ms
stage_pct  = [18.4, 78.3, 3.3]

# ── 繪圖 ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(C_BG)
fig.suptitle("LexiGaze v8 — 推理速度分析", fontsize=16, color=C_TEXT,
             fontweight='bold', y=1.01)

# ─── [左] process_file v7 vs v8 ──────────────────────────────────
ax1 = axes[0]
style_ax(ax1)
colors = [C_V7, C_V8]
bars = ax1.bar(pf_labels, pf_times, width=0.45, color=colors, alpha=0.88)
for bar, val in zip(bars, pf_times):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 12,
             f"{val:.0f} ms", ha='center', va='bottom',
             fontsize=13, color=C_TEXT, fontweight='bold')

# 加速比標記
speedup = pf_times[0] / pf_times[1]
ax1.annotate(f"⬇ {speedup:.1f}× 加速", xy=(1, pf_times[1]),
             xytext=(0.5, pf_times[0] * 0.55),
             fontsize=13, color="#FFD700", fontweight='bold',
             ha='center',
             arrowprops=dict(arrowstyle='->', color='#FFD700', lw=1.8))

ax1.set_ylim(0, max(pf_times) * 1.25)
ax1.set_ylabel("總耗時 (ms)", fontsize=11)
ax1.set_title("process_file 速度比較\n（25句文件, ~330詞）", fontsize=12, pad=10)

v7_patch = mpatches.Patch(color=C_V7, label='v7 逐句 run()')
v8_patch = mpatches.Patch(color=C_V8, label='v8 chunked')
ax1.legend(handles=[v7_patch, v8_patch], fontsize=9,
           facecolor=C_AX, labelcolor=C_TEXT, framealpha=0.7)

# ─── [中] 不同長度 per-word 延遲 ─────────────────────────────────
ax2 = axes[1]
style_ax(ax2)
bar_colors = ["#4C9BE8", "#F4A44A", "#6ABF69"]
bars2 = ax2.bar(len_labels, per_word, width=0.45, color=bar_colors, alpha=0.88)
for bar, val, total in zip(bars2, per_word, len_ms):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
             f"{val:.2f} ms/詞\n(共 {total:.0f}ms)",
             ha='center', va='bottom', fontsize=10, color=C_TEXT, fontweight='bold')

ax2.set_ylim(0, max(per_word) * 1.4)
ax2.set_ylabel("每詞平均耗時 (ms/word)", fontsize=11)
ax2.set_title("不同文本長度 per-word 延遲\n（v8 run()）", fontsize=12, pad=10)

# ─── [右] 各階段耗時拆解（Medium 文本）──────────────────────────
ax3 = axes[2]
style_ax(ax3)
stage_colors = [C_SPACY, C_GPT2, C_SCORE]
bars3 = ax3.bar(stages, stage_ms, width=0.45, color=stage_colors, alpha=0.88)
for bar, val, pct in zip(bars3, stage_ms, stage_pct):
    ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
             f"{val:.1f} ms\n({pct:.1f}%)",
             ha='center', va='bottom', fontsize=11, color=C_TEXT, fontweight='bold')

ax3.set_ylim(0, max(stage_ms) * 1.35)
ax3.set_ylabel("耗時 (ms)", fontsize=11)
ax3.set_title("單次推理各階段耗時拆解\n（Medium 文本, ~65詞）", fontsize=12, pad=10)

# GPT-2 瓶頸標記
ax3.annotate("⚠ 瓶頸", xy=(1, stage_ms[1]),
             xytext=(1.35, stage_ms[1] * 0.75),
             fontsize=11, color="#FFD700", fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#FFD700', lw=1.5))

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speed_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"[完成] 速度圖表已儲存至: {out_path}")
