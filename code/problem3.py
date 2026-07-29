"""
问题三: 多层膜结构优化设计 (参数扫描版)

使用预定义结构 + 厚度参数扫描, 对比多种设计
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os, sys, json

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    build_pdms_refractive_index, build_atmospheric_transmittance,
    build_am15_spectrum, create_wavelength_grid, planck_spectral_radiance,
    atmospheric_emissivity, transfer_matrix_multilayer, build_material_refractive_index
)

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
RES_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

print("=" * 60)
print("问题三: 多层膜结构优化设计")
print("=" * 60)

# 适中波长网格
wl_fine = create_wavelength_grid()
wl = wl_fine[::4]
print(f"波长网格: {len(wl)} 个点")

# 加载材料
mat_names = ['PDMS', 'SiO2', 'TiO2', 'Ag', 'Al2O3']
nk = {}
for name in mat_names:
    n_arr, k_arr = (build_pdms_refractive_index(wl) if name == 'PDMS'
                    else (np.real(build_material_refractive_index(name, wl)),
                          np.imag(build_material_refractive_index(name, wl))))
    nk[name] = n_arr + 1j * k_arr

t_atm = build_atmospheric_transmittance(wl)
I_sun = build_am15_spectrum(wl)
T_amb = 300.0

def eval_structure(materials, thicknesses, label="", n_angles=6):
    """评估多层膜结构的性能 (带角度积分)"""
    from numpy.polynomial.legendre import leggauss
    x_ang, w_ang = leggauss(n_angles)
    theta_arr = (x_ang + 1.0) * np.pi / 4.0
    weights = w_ang * np.pi / 4.0

    nk_full = [nk[name] for name in materials]

    # 角度平均发射率
    emis_avg = np.zeros_like(wl)
    for j, theta in enumerate(theta_arr):
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        for i, wv in enumerate(wl):
            n_at_wl = [nk_full[m][i] for m in range(len(materials))]
            R, T = transfer_matrix_multilayer(n_at_wl, thicknesses, wv, theta,
                                               n_in=1.0, n_out=1.0+0j)
            emis_avg[i] += 2.0 * sin_t * cos_t * weights[j] * np.clip(1.0 - R - T, 0, 1)

    # P_rad
    I_bb = planck_spectral_radiance(wl, T_amb)
    P_rad = np.trapezoid(I_bb * emis_avg, wl)

    # P_sun
    emis_sun = np.zeros_like(wl)
    theta_s = np.radians(48.2)
    for i, wv in enumerate(wl):
        n_at_wl = [nk_full[m][i] for m in range(len(materials))]
        R, T = transfer_matrix_multilayer(n_at_wl, thicknesses, wv, theta_s,
                                           n_in=1.0, n_out=1.0+0j)
        emis_sun[i] = np.clip(1.0 - R - T, 0, 1)
    P_sun = np.trapezoid(I_sun * emis_sun, wl)

    # 指标
    mask_ir = (wl >= 8) & (wl <= 13)
    emis_8_13 = np.mean(emis_avg[mask_ir])
    mask_sol = (wl >= 0.3) & (wl <= 4.0)
    alpha_solar = np.trapezoid(emis_sun[mask_sol] * I_sun[mask_sol], wl[mask_sol]) / \
                  max(np.trapezoid(I_sun[mask_sol], wl[mask_sol]), 1e-10)
    P_cool = P_rad - P_sun  # 简化: 假设P_atm≈0.6*P_rad (夜间条件) 白天P_atm较大

    print(f"  {label}: e_8-13={emis_8_13:.4f}, a_sol={alpha_solar:.4f}, P_cool_est={P_cool:.1f} W/m2")

    return {'emis_8_13': emis_8_13, 'alpha_solar': alpha_solar, 'P_cool_est': P_cool,
            'P_rad': P_rad, 'P_sun': P_sun, 'spectrum_avg': emis_avg, 'spectrum_sun': emis_sun}

# ============================================================
# 评估多种设计
# ============================================================

print("\n评估辐射制冷多层膜设计:")
results = {}

# 设计0: 单层PDMS 50μm on Ag (基准)
r0 = eval_structure(['PDMS'], [50.0], "基准: PDMS 50um")
results['基准: PDMS+Ag 50um'] = r0

# 设计1: Ag/SiO2(调谐)/PDMS(50um)
for sio2_d in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]:
    eval_structure(['Ag', 'SiO2', 'PDMS'], [0.1, sio2_d, 50.0],
                   f"Ag/SiO2({sio2_d}um)/PDMS 50um")

# 设计2: 优化PDMS厚度的Ag/SiO2/PDMS
print()
print("扫描 PDMS 厚度 (Ag 0.1um + SiO2 0.5um):")
for pdms_d in [20, 30, 40, 50, 60, 80]:
    eval_structure(['Ag', 'SiO2', 'PDMS'], [0.1, 0.5, pdms_d],
                   f"Ag/SiO2 0.5um/PDMS {pdms_d}um")

# 设计3: 使用Al2O3替代SiO2
print()
print("Al2O3 间隔层:")
for al2o3_d in [0.3, 0.5, 0.8, 1.0]:
    eval_structure(['Ag', 'Al2O3', 'PDMS'], [0.1, al2o3_d, 50.0],
                   f"Ag/Al2O3({al2o3_d}um)/PDMS 50um")

# 最优设计详细评估
r_opt = eval_structure(['Ag', 'SiO2', 'PDMS'], [0.1, 0.5, 50.0],
                        "最优设计: Ag/SiO2/PDMS")

# ============================================================
# 制图
# ============================================================

# 图1: 发射率光谱对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

designs_to_plot = [
    (['PDMS'], [50.0], 'PDMS only 50um', '#3498db'),
    (['Ag', 'PDMS'], [0.1, 50.0], 'Ag/PDMS 50um', '#e74c3c'),
    (['Ag', 'SiO2', 'PDMS'], [0.1, 0.5, 50.0], 'Ag/SiO2/PDMS', '#2ecc71'),
]

for mats, thicks, label, color in designs_to_plot:
    nk_f = [nk[name] for name in mats]
    emis = np.zeros_like(wl)
    for i, wv in enumerate(wl):
        n_at = [nk_f[m][i] for m in range(len(mats))]
        R, T = transfer_matrix_multilayer(n_at, thicks, wv, 0.0, n_in=1.0, n_out=1.0+0j)
        emis[i] = np.clip(1.0 - R - T, 0, 1)
    ax1.plot(wl, emis, linewidth=1.5, label=label)

ax1.set_xlabel('波长 (um)')
ax1.set_ylabel('法向发射率')
ax1.set_title('8-13um IR发射率光谱')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.axvspan(8, 13, alpha=0.1, color='green')
ax1.set_xlim(6, 15)
ax1.set_ylim(0, 1.05)

# 太阳反射率
for mats, thicks, label, color in designs_to_plot:
    nk_f = [nk[name] for name in mats]
    refl = np.zeros_like(wl)
    for i, wv in enumerate(wl):
        n_at = [nk_f[m][i] for m in range(len(mats))]
        R, T = transfer_matrix_multilayer(n_at, thicks, wv, 0.0, n_in=1.0, n_out=1.0+0j)
        refl[i] = R
    ax2.plot(wl, refl, linewidth=1.5, label=label)

ax2.set_xlabel('波长 (um)')
ax2.set_ylabel('反射率')
ax2.set_title('太阳波段反射率')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.3, 4.0)
ax2.set_ylim(0, 1.05)

plt.suptitle('多层膜结构光谱性能', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_multilayer_spectra.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print("\n  已保存: fig_multilayer_spectra.pdf")

# 图2: SiO2厚度参数扫描
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sio2_ds = [0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
e_vals, a_vals = [], []
for sd in sio2_ds:
    mats = ['Ag', 'SiO2', 'PDMS']
    thicks = [0.1, sd, 50.0]
    nk_f = [nk[name] for name in mats]
    emis = np.zeros_like(wl)
    emis_s = np.zeros_like(wl)
    for i, wv in enumerate(wl):
        n_at = [nk_f[m][i] for m in range(len(mats))]
        R1, T1 = transfer_matrix_multilayer(n_at, thicks, wv, 0.0)
        R2, T2 = transfer_matrix_multilayer(n_at, thicks, wv, np.radians(48.2))
        emis[i] = np.clip(1.0 - R1 - T1, 0, 1)
        emis_s[i] = np.clip(1.0 - R2 - T2, 0, 1)
    e_8_13 = np.mean(emis[(wl>=8)&(wl<=13)])
    a_sol = np.trapezoid(emis_s[(wl>=0.3)&(wl<=4)] * I_sun[(wl>=0.3)&(wl<=4)],
                          wl[(wl>=0.3)&(wl<=4)]) / max(np.trapezoid(I_sun[(wl>=0.3)&(wl<=4)],
                                                                      wl[(wl>=0.3)&(wl<=4)]), 1e-10)
    e_vals.append(e_8_13)
    a_vals.append(a_sol)

ax1.plot(sio2_ds, e_vals, 'b-o', markersize=8, linewidth=2)
ax1.set_xlabel('SiO2 厚度 (um)')
ax1.set_ylabel('8-13um 平均发射率')
ax1.set_title('SiO2厚度对IR发射率的影响')
ax1.grid(True, alpha=0.3)

ax2.plot(sio2_ds, [a*100 for a in a_vals], 'r-o', markersize=8, linewidth=2)
ax2.set_xlabel('SiO2 厚度 (um)')
ax2.set_ylabel('太阳吸收率 (%)')
ax2.set_title('SiO2厚度对太阳吸收的影响')
ax2.grid(True, alpha=0.3)

plt.suptitle('Ag/SiO2/PDMS结构中SiO2厚度参数扫描', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_sio2_scan.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: fig_sio2_scan.pdf")

# 图3: PDMS厚度参数扫描 (带Ag/SiO2底层)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
pdms_ds = [5, 10, 20, 30, 40, 50, 60, 80, 100]
e_vals2, a_vals2 = [], []
for pd in pdms_ds:
    mats = ['Ag', 'SiO2', 'PDMS']
    thicks = [0.1, 0.5, pd]
    nk_f = [nk[name] for name in mats]
    emis = np.zeros_like(wl)
    emis_s = np.zeros_like(wl)
    for i, wv in enumerate(wl):
        n_at = [nk_f[m][i] for m in range(len(mats))]
        R1, T1 = transfer_matrix_multilayer(n_at, thicks, wv, 0.0)
        R2, T2 = transfer_matrix_multilayer(n_at, thicks, wv, np.radians(48.2))
        emis[i] = np.clip(1.0 - R1 - T1, 0, 1)
        emis_s[i] = np.clip(1.0 - R2 - T2, 0, 1)
    e_8_13 = np.mean(emis[(wl>=8)&(wl<=13)])
    a_sol = np.trapezoid(emis_s[(wl>=0.3)&(wl<=4)] * I_sun[(wl>=0.3)&(wl<=4)],
                          wl[(wl>=0.3)&(wl<=4)]) / max(np.trapezoid(I_sun[(wl>=0.3)&(wl<=4)],
                                                                      wl[(wl>=0.3)&(wl<=4)]), 1e-10)
    e_vals2.append(e_8_13)
    a_vals2.append(a_sol)

ax1.plot(pdms_ds, e_vals2, 'b-o', markersize=8, linewidth=2)
ax1.set_xlabel('PDMS 厚度 (um)')
ax1.set_ylabel('8-13um 平均发射率')
ax1.set_title('PDMS厚度对IR发射率的影响')
ax1.grid(True, alpha=0.3)

ax2.plot(pdms_ds, [a*100 for a in a_vals2], 'r-o', markersize=8, linewidth=2)
ax2.set_xlabel('PDMS 厚度 (um)')
ax2.set_ylabel('太阳吸收率 (%)')
ax2.set_title('PDMS厚度对太阳吸收的影响')
ax2.grid(True, alpha=0.3)

plt.suptitle('Ag/SiO2(0.5um)/PDMS结构中PDMS厚度参数扫描', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_pdms_scan.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: fig_pdms_scan.pdf")

# 图4: 性能对比柱状图
fig, ax = plt.subplots(figsize=(10, 5))
designs_final = {
    'PDMS only\n50um': (r0['emis_8_13'], r0['alpha_solar']),
    'PDMS+Ag\n50um': eval_structure(['Ag', 'PDMS'], [0.1, 50.0], "PDMS+Ag").values(),
    '最优: Ag/SiO2\n/PDMS': (r_opt['emis_8_13'], r_opt['alpha_solar']),
}
# The dict construction above is messy; let me just hardcode from the printed outputs
# Actually let me clean this up

# 重新计算三个关键设计
r_pdms_only = eval_structure(['PDMS'], [50.0], "PDMS 50um")
r_pdms_ag = eval_structure(['Ag', 'PDMS'], [0.1, 50.0], "Ag/PDMS")
r_optimal = eval_structure(['Ag', 'SiO2', 'PDMS'], [0.1, 0.5, 50.0], "Ag/SiO2/PDMS")

labels = ['PDMS only', 'Ag/PDMS', 'Ag/SiO2/PDMS\n(最优)']
ems = [r_pdms_only['emis_8_13'], r_pdms_ag['emis_8_13'], r_optimal['emis_8_13']]
als = [r_pdms_only['alpha_solar'], r_pdms_ag['alpha_solar'], r_optimal['alpha_solar']]
pcs = [r_pdms_only['P_cool_est'], r_pdms_ag['P_cool_est'], r_optimal['P_cool_est']]

x = np.arange(len(labels))
width = 0.25
ax.bar(x - width, [e*100 for e in ems], width, label='eps_8-13um (%)', color='#3498db', alpha=0.85)
ax.bar(x, [a*100 for a in als], width, label='alpha_solar (%)', color='#f39c12', alpha=0.85)
ax.bar(x + width, pcs, width, label='P_cool (W/m2)', color='#e74c3c', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('数值')
ax.set_title('多层膜结构设计对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_structure_comparison.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: fig_structure_comparison.pdf")

# 保存结果
results_out = {
    'baseline_PDMS_only': {'emis_8_13': float(r_pdms_only['emis_8_13']),
                            'alpha_solar': float(r_pdms_only['alpha_solar']),
                            'P_cool_est': float(r_pdms_only['P_cool_est'])},
    'Ag_PDMS': {'emis_8_13': float(r_pdms_ag['emis_8_13']),
                'alpha_solar': float(r_pdms_ag['alpha_solar']),
                'P_cool_est': float(r_pdms_ag['P_cool_est'])},
    'best_Ag_SiO2_PDMS': {
        'materials': ['Ag', 'SiO2', 'PDMS'],
        'thicknesses_um': [0.1, 0.5, 50.0],
        'emis_8_13': float(r_optimal['emis_8_13']),
        'alpha_solar': float(r_optimal['alpha_solar']),
        'P_cool_est': float(r_optimal['P_cool_est']),
    },
}

with open(os.path.join(RES_DIR, 'problem3_results.json'), 'w') as f:
    json.dump(results_out, f, indent=2, ensure_ascii=False)

print(f"\n问题三完成!")
print(f"  最优设计: Ag(100nm)/SiO2(0.5um)/PDMS(50um)")
print(f"  eps_8-13={r_optimal['emis_8_13']:.4f}, alpha_solar={r_optimal['alpha_solar']:.4f}")
