"""
问题一: PDMS薄膜发射率随波长和膜厚变化的数学模型

使用传输矩阵法(TMM)计算不同厚度的PDMS薄膜的:
1. 光谱发射率 ε(λ, d)
2. 8-13μm平均发射率 vs 厚度
3. 干涉效应分析
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os, sys, json

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    build_pdms_refractive_index, create_wavelength_grid,
    compute_emissivity_spectrum, compute_emissivity_angle_averaged,
    transfer_matrix_single_layer
)

# 中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 输出目录
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
RES_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ============================================================
# 1. 准备数据
# ============================================================

print("=" * 60)
print("问题一: PDMS薄膜发射率计算")
print("=" * 60)

# 波长网格
wl = create_wavelength_grid()
print(f"波长网格: {len(wl)} 个点, 范围 {wl[0]:.2f} - {wl[-1]:.2f} μm")

# PDMS折射率
n_pdms, k_pdms = build_pdms_refractive_index(wl)
print(f"PDMS折射率范围: n={n_pdms.min():.3f}-{n_pdms.max():.3f}, k={k_pdms.min():.6f}-{k_pdms.max():.3f}")

# 膜厚序列 (对数均匀)
d_values = np.array([0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500])
print(f"膜厚序列: {d_values} μm")

# ============================================================
# 2. 计算法向发射率光谱
# ============================================================

print("\n计算法向发射率光谱...")
emissivity_normal = {}  # d → ε(λ)

for d in d_values:
    emis = compute_emissivity_spectrum(d, wl, 0.0, n_pdms, k_pdms, n_sub=1.0+0j)
    emissivity_normal[d] = emis
    print(f"  d={d:4.0f} μm: ε_avg(8-13μm)={np.mean(emis[(wl>=8)&(wl<=13)]):.4f}")

# ============================================================
# 3. 计算角度平均发射率
# ============================================================

print("\n计算角度平均发射率...")
emissivity_avg_hemi = {}
for d in [0.5, 5, 20, 50, 100, 200]:
    emis_avg = compute_emissivity_angle_averaged(d, wl, n_pdms, k_pdms, n_sub=1.0+0j, n_angles=16)
    emissivity_avg_hemi[d] = emis_avg
    print(f"  d={d:4.0f} μm: ε_avg_hemi(8-13μm)={np.mean(emis_avg[(wl>=8)&(wl<=13)]):.4f}")

# ============================================================
# 4. 8-13μm 平均发射率 vs 厚度
# ============================================================

window_mask = (wl >= 8.0) & (wl <= 13.0)
d_fine = np.logspace(np.log10(0.1), np.log10(500), 80)
avg_emissivity_window = []

for d in d_fine:
    emis = compute_emissivity_spectrum(d, wl, 0.0, n_pdms, k_pdms, n_sub=1.0+0j)
    avg_emis = np.mean(emis[window_mask])
    avg_emissivity_window.append(avg_emis)

# 寻找最优厚度
idx_opt = np.argmax(avg_emissivity_window)
d_opt = d_fine[idx_opt]
emis_opt = avg_emissivity_window[idx_opt]
print(f"\n最优膜厚: d≈{d_opt:.1f} μm, ε_avg(8-13μm)={emis_opt:.4f}")

# ============================================================
# 5. 制图
# ============================================================

# --- 图1: PDMS折射率 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax1.plot(wl, n_pdms, 'b-', linewidth=1.2)
ax1.set_ylabel('折射率实部 n', fontsize=12)
ax1.set_ylim(0.5, 2.5)
ax1.grid(True, alpha=0.3)
ax1.axvspan(8, 13, alpha=0.15, color='green', label='8-13μm 大气窗口')
ax1.legend(fontsize=9)

ax2.semilogy(wl, k_pdms, 'r-', linewidth=1.2)
ax2.set_xlabel('波长 (μm)', fontsize=12)
ax2.set_ylabel('消光系数 k', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.axvspan(8, 13, alpha=0.15, color='green')

# 标注吸收峰
peaks = {7.9: 'Si-CH₃', 9.3: 'Si-O-Si', 12.5: 'Si-C'}
for wl_peak, label in peaks.items():
    idx_p = np.argmin(np.abs(wl - wl_peak))
    ax2.annotate(label, (wl_peak, k_pdms[idx_p]),
                 xytext=(0, 15), textcoords='offset points',
                 fontsize=8, ha='center',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

plt.suptitle('PDMS 复折射率光谱', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_pdms_refractive_index.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: fig_pdms_refractive_index.pdf")

# --- 图2: 不同厚度的发射率光谱 ---
fig, axes = plt.subplots(2, 1, figsize=(12, 9))

# 薄层 (0.1-10μm)
thin_d = [0.1, 0.5, 1, 2, 5, 10]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(thin_d)))
for i, d in enumerate(thin_d):
    axes[0].plot(wl, emissivity_normal[d], color=colors[i], linewidth=1.2,
                 label=f'd={d} μm')
axes[0].set_ylabel('发射率 ε', fontsize=12)
axes[0].set_title('薄层 PDMS', fontsize=12)
axes[0].legend(fontsize=8, ncol=3)
axes[0].grid(True, alpha=0.3)
axes[0].axvspan(8, 13, alpha=0.1, color='green')
axes[0].set_ylim(-0.02, 1.02)

# 厚层 (10-200μm)
thick_d = [10, 20, 40, 60, 100, 200]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(thick_d)))
for i, d in enumerate(thick_d):
    axes[1].plot(wl, emissivity_normal[d], color=colors[i], linewidth=1.2,
                 label=f'd={d} μm')
axes[1].set_xlabel('波长 (μm)', fontsize=12)
axes[1].set_ylabel('发射率 ε', fontsize=12)
axes[1].set_title('厚层 PDMS', fontsize=12)
axes[1].legend(fontsize=8, ncol=3)
axes[1].grid(True, alpha=0.3)
axes[1].axvspan(8, 13, alpha=0.1, color='green')
axes[1].set_ylim(-0.02, 1.02)

plt.suptitle('不同厚度 PDMS 薄膜的光谱发射率 ε(λ) — 法向入射', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_emissivity_spectrum.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: fig_emissivity_spectrum.pdf")

# --- 图3: 8-13μm平均发射率 vs 厚度 ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogx(d_fine, avg_emissivity_window, 'b-', linewidth=2)
ax.axvline(d_opt, color='red', linestyle='--', linewidth=1.2,
           label=f'最优厚度 d≈{d_opt:.1f} μm')
ax.axhline(emis_opt, color='red', linestyle=':', linewidth=1.2,
           label=f'最大平均发射率 ε≈{emis_opt:.4f}')
ax.set_xlabel('PDMS 膜厚 (μm)', fontsize=12)
ax.set_ylabel('8-13μm 平均发射率', fontsize=12)
ax.set_title('8-13μm 大气窗口平均发射率随膜厚的变化', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_avg_emissivity_vs_thickness.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: fig_avg_emissivity_vs_thickness.pdf")

# --- 图4: 法向vs角度平均发射率对比 ---
fig, ax = plt.subplots(figsize=(10, 5))
d_comp = 50.0
emis_normal_50 = emissivity_normal[50.0]
emis_avg_50 = emissivity_avg_hemi[50.0]
ax.plot(wl, emis_normal_50, 'b-', linewidth=1.2, alpha=0.7, label=f'法向发射率 (d=50μm)')
ax.plot(wl, emis_avg_50, 'r--', linewidth=1.2, alpha=0.7, label=f'角度平均发射率 (d=50μm)')
ax.fill_between(wl[(wl>=8)&(wl<=13)], 0, 1, alpha=0.08, color='green')
ax.text(10.5, 0.95, '8-13μm 大气窗口', ha='center', fontsize=10, color='green')
ax.set_xlabel('波长 (μm)', fontsize=12)
ax.set_ylabel('发射率 ε', fontsize=12)
ax.set_title('法向发射率与角度平均发射率对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_emissivity_normal_vs_avg.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: fig_emissivity_normal_vs_avg.pdf")

# --- 图5: 发射率热力图 ε(λ, d) ---
fig, ax = plt.subplots(figsize=(10, 6))
d_log = np.logspace(np.log10(0.1), np.log10(300), 40)
E_matrix = np.zeros((len(d_log), len(wl)))
for i, d in enumerate(d_log):
    E_matrix[i, :] = compute_emissivity_spectrum(d, wl, 0.0, n_pdms, k_pdms, n_sub=1.0+0j)

im = ax.pcolormesh(wl, d_log, E_matrix, shading='auto', cmap='hot',
                    vmin=0, vmax=1)
ax.set_yscale('log')
ax.set_xlabel('波长 (μm)', fontsize=12)
ax.set_ylabel('膜厚 d (μm)', fontsize=12)
ax.set_title('PDMS 薄膜发射率 ε(λ, d) 热力图', fontsize=14, fontweight='bold')
ax.axvspan(8, 13, alpha=0.15, color='cyan')
ax.text(10.5, 250, '大气窗口', ha='center', fontsize=9, color='cyan')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('发射率 ε', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_emissivity_heatmap.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: fig_emissivity_heatmap.pdf")

# ============================================================
# 6. 保存结果数据
# ============================================================

results = {
    'd_opt_micron': float(d_opt),
    'max_avg_emissivity_8_13um': float(emis_opt),
    'd_values_micron': d_fine.tolist(),
    'avg_emissivity_window': avg_emissivity_window,
    'wavelength_grid': wl.tolist(),
    'emissivity_at_50um': emissivity_normal[50.0].tolist(),
}

# 保存每个关键厚度的8-13μm平均发射率
for d_key in [5, 10, 20, 30, 40, 50, 60, 80, 100]:
    emis_d = emissivity_normal[float(d_key)] if float(d_key) in emissivity_normal else None
    if emis_d is not None:
        results[f'avg_emissivity_d{d_key}um'] = float(np.mean(emis_d[window_mask]))

with open(os.path.join(RES_DIR, 'problem1_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# 保存关键厚度发射率数据为CSV
header = 'wavelength_um,' + ','.join([f'emissivity_d{d}um' for d in [0.5, 5, 20, 50, 100, 200]])
data = np.column_stack([wl] + [emissivity_normal[d] for d in [0.5, 5, 20, 50, 100, 200]])
np.savetxt(os.path.join(RES_DIR, 'emissivity_spectra.csv'), data,
           delimiter=',', header=header, comments='', fmt='%.6f')

print("\n问题一完成! 结果已保存到 results/ 和 figures/")
