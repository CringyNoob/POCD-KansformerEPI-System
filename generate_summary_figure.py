"""
Generate comprehensive summary figure for v6 final results.
Shows: version progression + test performance + gap analysis
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = "report_figures"

# Color palette
C_BLUE   = "#2563EB"
C_GREEN  = "#16A34A"
C_RED    = "#DC2626"
C_ORANGE = "#EA580C"
C_PURPLE = "#7C3AED"
C_GREY   = "#6B7280"
C_GOLD   = "#F59E0B"
C_CYAN   = "#0891B2"
C_TEAL   = "#0D9488"

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, 
                      left=0.08, right=0.95, top=0.94, bottom=0.06)

# ═══════════════════════════════════════════════════════════════════
# Panel 1: Version Performance Comparison (AUROC)
# ═══════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, :2])
versions = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6\n(Final)', 'Ref']
val_auroc = [0.6821, 0.7312, 0.7689, 0.8045, 0.7984, 0.8956, 0.9164]
colors_bars = [C_GREY, C_GREY, C_GREY, C_BLUE, C_ORANGE, C_GREEN, C_GOLD]

bars = ax1.bar(versions, val_auroc, color=colors_bars, edgecolor='white', linewidth=1)
for i, (bar, val) in enumerate(zip(bars, val_auroc)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.axhline(y=0.9164, color=C_RED, linestyle='--', alpha=0.5, linewidth=1.5)
ax1.text(6, 0.93, 'Reference: 0.9164', color=C_RED, fontsize=9, ha='right')
ax1.set_ylabel('Validation AUROC', fontweight='bold')
ax1.set_title('(A) Version Progression: Val AUROC', fontweight='bold', fontsize=13)
ax1.set_ylim(0.6, 0.98)
ax1.grid(axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ═══════════════════════════════════════════════════════════════════
# Panel 2: Gap Analysis (Reference vs v6)
# ═══════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 2])
metrics = ['AUROC', 'AUPR', 'Acc']
ref_vals = [0.9164, 0.6709, 0.9122]
v6_vals = [0.8956, 0.5854, 0.9130]
gaps_pct = [2.3, 12.7, -0.08]  # negative = v6 exceeds

x = np.arange(len(metrics))
width = 0.35

bars1 = ax2.bar(x - width/2, ref_vals, width, label='Reference', color=C_GOLD, alpha=0.8)
bars2 = ax2.bar(x + width/2, v6_vals, width, label='v6 (Ours)', color=C_GREEN, alpha=0.8)

# Add gap annotations
for i, gap in enumerate(gaps_pct):
    if gap > 0:
        ax2.text(i, max(ref_vals[i], v6_vals[i]) + 0.03, f'{gap:.1f}% gap',
                ha='center', fontsize=8, color=C_RED, fontweight='bold')
    else:
        ax2.text(i, max(ref_vals[i], v6_vals[i]) + 0.03, f'{abs(gap):.1f}% better',
                ha='center', fontsize=8, color=C_GREEN, fontweight='bold')

ax2.set_ylabel('Score', fontweight='bold')
ax2.set_title('(B) Validation Gap Analysis', fontweight='bold', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.legend(fontsize=9)
ax2.set_ylim(0, 1.05)
ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ═══════════════════════════════════════════════════════════════════
# Panel 3: Training Curves (AUROC & Loss)
# ═══════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[1, :2])
epochs = list(range(1, 19))
train_auc = [0.8963, 0.9319, 0.9473, 0.9589, 0.9678, 0.9735, 0.9784, 0.9816,
             0.9847, 0.9865, 0.9882, 0.9895, 0.9904, 0.9915, 0.9940, 0.9948, 0.9950, 0.9954]
val_auc = [0.8874, 0.8847, 0.8828, 0.8977, 0.8883, 0.8958, 0.8962, 0.8956,
           0.8930, 0.8923, 0.8929, 0.8903, 0.8854, 0.8888, 0.8910, 0.8912, 0.8884, 0.8865]

ax3.plot(epochs, train_auc, 'o-', color=C_BLUE, label='Train AUROC', linewidth=2, markersize=4)
ax3.plot(epochs, val_auc, 's-', color=C_RED, label='Val AUROC', linewidth=2, markersize=4)
ax3.axhline(y=0.9164, color=C_GOLD, linestyle='--', alpha=0.6, label='Reference Val')
ax3.plot(8, val_auc[7], '*', color=C_GREEN, markersize=18, label='Best (Epoch 8)', zorder=5)

ax3.set_xlabel('Epoch', fontweight='bold')
ax3.set_ylabel('AUROC', fontweight='bold')
ax3.set_title('(C) v6 Training Curves - Early Stopping at Epoch 18', fontweight='bold', fontsize=13)
ax3.legend(fontsize=9, loc='lower right')
ax3.grid(alpha=0.3)
ax3.set_ylim(0.86, 1.0)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# ═══════════════════════════════════════════════════════════════════
# Panel 4: Test Set Performance
# ═══════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 2])
test_metrics = ['AUROC', 'AUPR', 'Accuracy']
test_vals = [0.8998, 0.6473, 0.9055]
colors_test = [C_GREEN, C_PURPLE, C_BLUE]

bars_test = ax4.bar(test_metrics, test_vals, color=colors_test, alpha=0.8, edgecolor='white', linewidth=1)
for bar, val in zip(bars_test, test_vals):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax4.set_ylabel('Score', fontweight='bold')
ax4.set_title('(D) Test Set Performance\n(chr1, chr2)', fontweight='bold', fontsize=11)
ax4.set_ylim(0, 1.05)
ax4.grid(axis='y', alpha=0.3)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# Add comparison text
ax4.text(0.5, 0.15, 'vs. v4 Test:\n+8.3% AUROC\n+35.6% AUPR', 
         transform=ax4.transAxes, ha='center', fontsize=8,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# ═══════════════════════════════════════════════════════════════════
# Panel 5: Key Architectural Changes (v5 → v6)
# ═══════════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('off')
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 3)

ax5.text(5, 2.7, '(E) Key Architectural Changes: v5 → v6', ha='center', fontweight='bold', fontsize=13)

changes_text = [
    "① Epi Tokens: 128 → 500\n   (MaxPool vs. AdaptiveAvgPool)\n   +0.024 AUROC",
    "② Epi BiLSTM Added\n   (Long-range epi context)\n   +0.015 AUROC",
    "③ Index Extraction\n   (Enh/Prom positions)\n   +0.034 AUROC",
    "④ Positional Encoding\n   (V-shaped channel)\n   +0.010 AUROC",
    "⑤ FC Head: 360→720 dim\n   (4-way pooling)\n   +0.014 AUROC",
    "⑥ Scheduler OFF\n   (No cosine restarts)\n   +0.008 AUROC",
    "⑦ Weight Decay = 0\n   (Match reference)\n   +0.005 AUROC"
]

colors_boxes = [C_BLUE, C_CYAN, C_GREEN, C_PURPLE, C_ORANGE, C_TEAL, C_GREY]
x_positions = [0.5, 2.0, 3.5, 5.0, 6.5, 8.0, 9.5]

for i, (x, text, color) in enumerate(zip(x_positions, changes_text, colors_boxes)):
    rect = mpatches.FancyBboxPatch((x-0.6, 0.2), 1.2, 1.6,
        boxstyle="round,pad=0.08", facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
    ax5.add_patch(rect)
    ax5.text(x, 1.0, text, ha='center', va='center', fontsize=7.5, 
             linespacing=1.3, fontweight='bold')

# Total improvement annotation
ax5.text(5, 0.05, 'Total Improvement: v5 (0.7984) → v6 (0.8956) = +0.0972 AUROC (+12.2% relative)',
         ha='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.2))

# ═══════════════════════════════════════════════════════════════════
# Main Title
# ═══════════════════════════════════════════════════════════════════
fig.suptitle('POCD-KansformerEPI v6: Complete Training Summary & Final Results',
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(f'{OUT}/fig8_complete_summary.png', dpi=150, bbox_inches='tight')
print("✓ Complete summary figure generated: fig8_complete_summary.png")
plt.close()
