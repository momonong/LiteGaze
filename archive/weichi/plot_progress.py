"""
繪製 LexiGaze Pipeline 各版本效能進度圖（v1–v6）
雙 Benchmark：學術文本（built-in）+ GECO 小說語料
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.rcParams['font.family'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 學術文本 Benchmark（built-in 3 句，9 GT 詞）──────────────────
acad_versions = [
    "v1\nBERT",
    "v2\nGPT-2\n(初版)",
    "v3\nGPT-2\n(70th%ile\n+縮寫過濾)",
    "v4\nGPT-2\n(搭配詞\nBoost)",
    "v5\n+AoA\n特徵",
    "v6\nRidge\n(GECO訓練)",
    "v7\nDomain\nAdaptive",
    "v8\nADJ權重\n+Boost修正",
]
acad_precision = [10.5, 27.3, 88.9, 90.0, 90.0, 75.0, 66.7, 88.9]
acad_recall    = [22.2, 66.7, 88.9, 100.0, 100.0, 66.7, 88.9, 88.9]
acad_f1        = [14.3, 38.7, 88.9, 94.7, 94.7, 70.6, 76.2, 88.9]
acad_hits      = [2, 6, 8, 9, 9, 6, 8, 8]
acad_missed    = [7, 3, 1, 0, 0, 3, 1, 1]
acad_gt        = 9

acad_notes = [
    "BERT 雙向模型\n(surprisal 低估)",
    "切換 GPT-2\n+ROOT 負載\n+全文門檻",
    "門檻 75→70th\n+縮寫過濾\nfingerprints 補回",
    "搭配詞 Boost\n(paradigm→shift)\n+zipf≤5 防誤報\nRecall 100%",
    "Kuperman AoA\n特徵加入\n(學術文 不變)",
    "Ridge Regression\n在 GECO 訓練\n(學術文退步\n小說進步)",
    "Domain 偵測\n學術→v5權重\nRidge回升+5.6%",
    "ADJ surprisal↓\nAoA↑+Boost近接\n+ADJ關卡\nPrec 88.9%",
]

# ── GECO Benchmark（Christie 小說前 5 句，17 GT 詞）──────────────
geco_versions = [
    "v4\nGPT-2\n+Boost",
    "v5\n+AoA",
    "v6\nRidge",
    "v7\nDomain\nAdaptive",
    "v8\nADJ修正",
]
geco_precision = [48.4, 48.4, 56.0, 56.0, 56.0]
geco_recall    = [88.2, 88.2, 82.4, 82.4, 82.4]
geco_f1        = [62.5, 62.5, 66.7, 66.7, 66.7]
geco_hits      = [15, 15, 14, 14, 14]
geco_missed    = [2, 2, 3, 3, 3]
geco_gt        = 17

# ── 色彩 ─────────────────────────────────────────────────────────
C_PREC = "#4C9BE8"
C_REC  = "#F4A44A"
C_F1   = "#6ABF69"
C_HIT  = "#6ABF69"
C_MISS = "#E05C5C"
C_BG   = "#1E1E2E"
C_AX   = "#2A2A3E"
C_SP   = "#555577"

def style_ax(ax):
    ax.set_facecolor(C_AX)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for sp in ax.spines.values():
        sp.set_edgecolor(C_SP)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='white')
    ax.set_axisbelow(True)

fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor(C_BG)
fig.suptitle("LexiGaze Cognitive Load Pipeline — 版本效能進展 (v1–v8)",
             fontsize=16, color='white', fontweight='bold', y=1.01)

# 2 列 × 3 欄 grid
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

# ─── [0,0] 學術文本 P/R/F1 折線 + 長條 ──────────────────────────
ax1 = fig.add_subplot(gs[0, :2])   # 跨兩欄
style_ax(ax1)
x = np.arange(len(acad_versions))
w = 0.22
bars_p = ax1.bar(x - w, acad_precision, w, label='Precision', color=C_PREC, alpha=0.85)
bars_r = ax1.bar(x,     acad_recall,    w, label='Recall',    color=C_REC,  alpha=0.85)
bars_f = ax1.bar(x + w, acad_f1,        w, label='F1 Score',  color=C_F1,   alpha=0.85)
for bars in [bars_p, bars_r, bars_f]:
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                 f"{h:.1f}", ha='center', va='bottom', fontsize=7.5, color='white', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(acad_versions, fontsize=8.5, color='white')
ax1.set_ylim(0, 110)
ax1.set_ylabel("Score (%)", fontsize=10)
ax1.set_title("【學術文本 Benchmark】Precision / Recall / F1 各版本比較 (GT=9 詞)", fontsize=11, pad=10)
ax1.legend(fontsize=8, facecolor=C_AX, labelcolor='white', framealpha=0.6)
for i, note in enumerate(acad_notes):
    ax1.annotate(note, xy=(x[i], 4), ha='center', va='bottom',
                 fontsize=5.8, color='#AAAACC',
                 bbox=dict(boxstyle='round,pad=0.3', fc=C_BG, ec=C_SP, alpha=0.75))

# ─── [0,2] 學術文本 命中/漏失 ───────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
style_ax(ax2)
xa = np.arange(len(acad_versions))
ax2.bar(xa, acad_hits,   0.5, label=f'命中 (GT={acad_gt})', color=C_HIT,  alpha=0.85)
ax2.bar(xa, acad_missed, 0.5, bottom=acad_hits, label='漏失', color=C_MISS, alpha=0.85)
for i in range(len(acad_versions)):
    ax2.text(xa[i], acad_hits[i] / 2, str(acad_hits[i]),
             ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    if acad_missed[i] > 0:
        ax2.text(xa[i], acad_hits[i] + acad_missed[i] / 2, str(acad_missed[i]),
                 ha='center', va='center', fontsize=11, color='white', fontweight='bold')
ax2.set_xticks(xa)
ax2.set_xticklabels(acad_versions, fontsize=7.5, color='white')
ax2.set_ylim(0, acad_gt + 1.5)
ax2.set_yticks(range(acad_gt + 1))
ax2.set_ylabel(f"詞數 (GT={acad_gt})", fontsize=10)
ax2.set_title("學術文本\n命中 / 漏失", fontsize=11, pad=10)
ax2.legend(fontsize=8, facecolor=C_AX, labelcolor='white', framealpha=0.6)

# ─── [1,0] GECO P/R/F1 ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
style_ax(ax3)
xg = np.arange(len(geco_versions))
ax3.bar(xg - w, geco_precision, w, label='Precision', color=C_PREC, alpha=0.85)
ax3.bar(xg,     geco_recall,    w, label='Recall',    color=C_REC,  alpha=0.85)
bars_gf = ax3.bar(xg + w, geco_f1, w, label='F1 Score', color=C_F1, alpha=0.85)
for bars in [ax3.containers[0], ax3.containers[1], ax3.containers[2]]:
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                 f"{h:.1f}", ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')
ax3.set_xticks(xg)
ax3.set_xticklabels(geco_versions, fontsize=10, color='white')
ax3.set_ylim(0, 110)
ax3.set_ylabel("Score (%)", fontsize=10)
ax3.set_title("【GECO Benchmark（Christie 小說）】Precision / Recall / F1 (GT=17 詞, v4–v8)", fontsize=11, pad=10)
ax3.legend(fontsize=8, facecolor=C_AX, labelcolor='white', framealpha=0.6)
geco_notes = [
    "基準線\n(學術優化)",
    "加入 AoA\n(小說無改善)",
    "Ridge 在 GECO\n訓練後\nF1 +4.2%",
    "Domain 偵測\nGECO→general\n維持 Ridge",
    "ADJ修正\n(小說不受影響\nRidge 先跑)",
]
for i, note in enumerate(geco_notes):
    ax3.annotate(note, xy=(xg[i], 4), ha='center', va='bottom',
                 fontsize=7.5, color='#AAAACC',
                 bbox=dict(boxstyle='round,pad=0.3', fc=C_BG, ec=C_SP, alpha=0.75))

# ─── [1,2] GECO 命中/漏失 ────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
style_ax(ax4)
xgr = np.arange(len(geco_versions))
ax4.bar(xgr, geco_hits,   0.5, label=f'命中 (GT={geco_gt})', color=C_HIT,  alpha=0.85)
ax4.bar(xgr, geco_missed, 0.5, bottom=geco_hits, label='漏失', color=C_MISS, alpha=0.85)
for i in range(len(geco_versions)):
    ax4.text(xgr[i], geco_hits[i] / 2, str(geco_hits[i]),
             ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    if geco_missed[i] > 0:
        ax4.text(xgr[i], geco_hits[i] + geco_missed[i] / 2, str(geco_missed[i]),
                 ha='center', va='center', fontsize=12, color='white', fontweight='bold')
ax4.set_xticks(xgr)
ax4.set_xticklabels(geco_versions, fontsize=9, color='white')
ax4.set_ylim(0, geco_gt + 2)
ax4.set_yticks(range(0, geco_gt + 2, 2))
ax4.set_ylabel(f"詞數 (GT={geco_gt})", fontsize=10)
ax4.set_title("GECO\n命中 / 漏失", fontsize=11, pad=10)
ax4.legend(fontsize=8, facecolor=C_AX, labelcolor='white', framealpha=0.6)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_progress.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"[完成] 圖表已儲存至: {out_path}")
