"""
问题二: 不同厚度PDMS薄膜辐射制冷性能评估

评估指标:
1. 净制冷功率 P_cool
2. 稳态温度降 ΔT
3. 制冷效率 η
4. 8-13μm发射率与太阳吸收率
5. 综合评价与建议
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
    compute_net_cooling_power, find_equilibrium_temperature,
    build_am15_spectrum, build_atmospheric_transmittance,
    build_material_refractive_index
)

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
RES_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ============================================================
# 1. 准备数据
# ============================================================

print("=" * 60)
print("问题二: PDMS辐射制冷性能评估")
print("=" * 60)

# 使用较粗的波长网格以提高计算速度 (角度积分计算量大)
wl_fine = create_wavelength_grid()
wl = wl_fine[::2]  # 降采样加速
print(f"波长网格: {len(wl)} 个点")

n_pdms, k_pdms = build_pdms_refractive_index(wl)

# 银反射基底 (波长相关复折射率)
n_ag_sub = build_material_refractive_index('Ag', wl)

# 环境参数
T_amb = 300.0  # 环境温度 27°C
h_c = 6.9      # 非辐射热交换系数 W/(m²·K)
theta_sun = np.radians(48.2)  # AM1.5太阳天顶角

# 膜厚序列
d_values = np.array([1, 2, 5, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300])

# ============================================================
# 2. 对每个厚度计算制冷性能
# ============================================================

print("\n计算各厚度的制冷性能 (此步较慢, 约需1-3分钟)...")
print("(角度积分使用高斯-勒让德求积法, 每厚度约需计算12个角度 × 500+波长点)")

performance = []

for d in d_values:
    print(f"  计算 d={d:4.0f} μm ...", end=' ', flush=True)

    # 计算在环境温度下的净制冷功率
    result = compute_net_cooling_power(
        d, wl, n_pdms, k_pdms,
        T=T_amb, T_amb=T_amb, h_c=h_c,
        n_sub=n_ag_sub, theta_sun=theta_sun, n_angles=12
    )

    # 计算稳态温度降
    T_eq = find_equilibrium_temperature(
        d, wl, n_pdms, k_pdms,
        T_amb=T_amb, h_c=h_c, n_sub=n_ag_sub
    )
    delta_T = T_amb - T_eq

    # 8-13μm 平均发射率 (法向)
    emis_normal = compute_emissivity_spectrum(d, wl, 0.0, n_pdms, k_pdms, n_sub=n_ag_sub)
    window_mask = (wl >= 8.0) & (wl <= 13.0)
    emis_8_13 = np.mean(emis_normal[window_mask])

    # 太阳吸收率 (0.3-4μm)
    solar_mask = (wl >= 0.3) & (wl <= 4.0)
    emis_solar = compute_emissivity_spectrum(d, wl, theta_sun, n_pdms, k_pdms, n_sub=n_ag_sub)
    I_sun = build_am15_spectrum(wl)
    total_sun = np.trapezoid(I_sun[solar_mask], wl[solar_mask])
    if total_sun > 0:
        alpha_solar = np.trapezoid(emis_solar[solar_mask] * I_sun[solar_mask], wl[solar_mask]) / total_sun
    else:
        alpha_solar = 0.0

    # 效率
    P_sun_total = np.trapezoid(build_am15_spectrum(wl), wl)
    eta = result['P_cool'] / P_sun_total if P_sun_total > 0 else 0.0

    performance.append({
        'd': float(d),
        'P_rad': result['P_rad'],
        'P_atm': result['P_atm'],
        'P_sun': result['P_sun'],
        'P_nonrad': result['P_nonrad'],
        'P_cool': result['P_cool'],
        'delta_T': delta_T,
        'emis_8_13um': emis_8_13,
        'alpha_solar': alpha_solar,
        'eta': eta,
    })

    print(f"P_cool={result['P_cool']:.1f} W/m², ΔT={delta_T:.2f} K, ε_8-13={emis_8_13:.3f}")

# ============================================================
# 3. 找出最优厚度
# ============================================================

perf_sorted = sorted(performance, key=lambda x: x['P_cool'], reverse=True)
best = perf_sorted[0]
best_emis = max(performance, key=lambda x: x['emis_8_13um'])
best_dT = max(performance, key=lambda x: x['delta_T'])

print(f"\n最优厚度 (按P_cool): d={best['d']:.0f} μm, P_cool={best['P_cool']:.1f} W/m²")
print(f"最优发射率: d={best_emis['d']:.0f} μm, ε_8-13={best_emis['emis_8_13um']:.4f}")
print(f"最大温降: d={best_dT['d']:.0f} μm, ΔT={best_dT['delta_T']:.2f} K")

# ============================================================
# 4. 制图
# ============================================================

d_arr = np.array([p['d'] for p in performance])

# --- 图1: 净制冷功率及其分量 vs 厚度 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# 左: 辐射分量
P_rad_arr = np.array([p['P_rad'] for p in performance])
P_atm_arr = np.array([p['P_atm'] for p in performance])
P_sun_arr = np.array([p['P_sun'] for p in performance])
P_cool_arr = np.array([p['P_cool'] for p in performance])

ax1.semilogx(d_arr, P_rad_arr, 'r-o', markersize=6, linewidth=1.8, label='向外辐射 P_rad')
ax1.semilogx(d_arr, P_atm_arr, 'b-s', markersize=6, linewidth=1.8, label='大气逆辐射 P_atm')
ax1.semilogx(d_arr, P_sun_arr, 'orange', marker='^', markersize=6, linewidth=1.8, label='太阳吸收 P_sun')
ax1.semilogx(d_arr, P_cool_arr, 'g-D', markersize=7, linewidth=2.2, label='净制冷功率 P_cool')
ax1.set_xlabel('PDMS 膜厚 (μm)', fontsize=12)
ax1.set_ylabel('功率 (W/m²)', fontsize=12)
ax1.set_title('辐射制冷各分量随膜厚变化', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 右: 温降和效率
dT_arr = np.array([p['delta_T'] for p in performance])
eta_arr = np.array([p['eta'] for p in performance])

ax2_eta = ax2
ax2_dT = ax2.twinx()

line1, = ax2_eta.semilogx(d_arr, dT_arr, 'b-D', markersize=7, linewidth=2.2, label='温度降 ΔT (K)')
line2, = ax2_dT.semilogx(d_arr, eta_arr * 100, 'r-^', markersize=7, linewidth=2.2, label='制冷效率 η (%)')
ax2_eta.set_xlabel('PDMS 膜厚 (μm)', fontsize=12)
ax2_eta.set_ylabel('温度降 ΔT (K)', fontsize=12, color='b')
ax2_dT.set_ylabel('制冷效率 η (%)', fontsize=12, color='r')
ax2_eta.set_title('稳态温度降与制冷效率', fontsize=13, fontweight='bold')
ax2_eta.legend([line1, line2], [line1.get_label(), line2.get_label()], fontsize=9, loc='upper left')
ax2_eta.grid(True, alpha=0.3)

plt.suptitle('PDMS 薄膜辐射制冷性能评估', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_cooling_performance.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: fig_cooling_performance.pdf")

# --- 图2: 评价指标雷达图 ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5), subplot_kw={'projection': 'polar'})

# 选择代表性的厚度进行对比
d_compare = [20, 50, 100]

# 归一化指标
metrics_names = ['8-13μm发射率', '太阳反射率\n(1-α)', '净制冷功率', '温度降', '制冷效率']
metrics_keys = ['emis_8_13um', 'alpha_solar', 'P_cool', 'delta_T', 'eta']

# 收集所有指标以归一化
all_vals = {k: [] for k in metrics_keys}
for p in performance:
    if p['d'] in d_compare:
        for k in metrics_keys:
            if k == 'alpha_solar':
                all_vals[k].append(1.0 - p[k])  # 太阳反射率
            else:
                all_vals[k].append(p[k])

max_vals = {k: max(v) if v else 1.0 for k, v in all_vals.items()}

colors_radar = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

for idx, d in enumerate(d_compare):
    ax = axes[idx]
    p = next(pp for pp in performance if pp['d'] == d)

    # 处理各指标, 确保分母不为0
    def safe_div(a, b):
        return a / b if abs(b) > 1e-10 else 0.0

    values = [
        safe_div(p['emis_8_13um'], max_vals.get('emis_8_13um', 1)),
        safe_div(1.0 - p['alpha_solar'], max_vals.get('alpha_solar', 1)),
        safe_div(max(p['P_cool'], 0), max_vals.get('P_cool', 1)),  # 负值截断为0
        safe_div(max(p['delta_T'], 0), max_vals.get('delta_T', 1)),
        safe_div(max(p['eta'], 0), max_vals.get('eta', 1)),
    ]
    values.append(values[0])  # 闭合雷达图

    N = len(metrics_names)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.fill(angles, values, alpha=0.25, color=colors_radar[idx])
    ax.plot(angles, values, 'o-', linewidth=2, color=colors_radar[idx])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'd={d} μm', fontsize=12, fontweight='bold', pad=20)
    ax.set_rlabel_position(30)

plt.suptitle('不同厚度 PDMS 辐射制冷性能雷达图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_radar_performance.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: fig_radar_performance.pdf")

# --- 图3: 各辐射分量的光谱贡献 ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 对50μm厚度的详细分析
d_detail = 50.0
emis_50 = compute_emissivity_spectrum(d_detail, wl, 0.0, n_pdms, k_pdms, n_sub=n_ag_sub)
I_bb_300K = None
from utils import planck_spectral_radiance
I_bb = planck_spectral_radiance(wl, 300.0)

# 左: 发射率 + 黑体辐射
axes[0].fill_between(wl, 0, emis_50, alpha=0.3, color='red', label=f'发射率 ε (d={d_detail:.0f}μm)')
ax0b = axes[0].twinx()
ax0b.plot(wl, I_bb, 'b-', linewidth=1.2, alpha=0.7, label='黑体辐射 300K')
axes[0].set_xlabel('波长 (μm)', fontsize=11)
axes[0].set_ylabel('发射率 ε', fontsize=11, color='red')
ax0b.set_ylabel('黑体辐亮度 (W/(m²·μm·sr))', fontsize=11, color='blue')
axes[0].set_title('发射率与300K黑体辐射', fontsize=11)
axes[0].axvspan(8, 13, alpha=0.1, color='green')
lines0 = axes[0].get_legend_handles_labels()
lines0b = ax0b.get_legend_handles_labels()
axes[0].legend(lines0[0] + lines0b[0], lines0[1] + lines0b[1], fontsize=8)

# 中: 大气光谱
t_atm = build_atmospheric_transmittance(wl)
axes[1].plot(wl, t_atm, 'g-', linewidth=1.5)
axes[1].fill_between(wl, 0, 1, where=(wl>=8)&(wl<=13), alpha=0.15, color='green')
axes[1].set_xlabel('波长 (μm)', fontsize=11)
axes[1].set_ylabel('大气透过率', fontsize=11)
axes[1].set_title('大气透过率光谱 (8-13μm窗口)', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 1.05)

# 右: 太阳光谱及PDMS吸收
I_sun = build_am15_spectrum(wl)
alpha_solar_spec = emis_50  # α(λ) = ε(λ)
P_sun_spec = I_sun * alpha_solar_spec
axes[2].fill_between(wl, 0, I_sun, alpha=0.3, color='orange', label='AM1.5太阳光谱')
axes[2].fill_between(wl, 0, P_sun_spec, alpha=0.5, color='red', label='PDMS吸收')
axes[2].set_xlabel('波长 (μm)', fontsize=11)
axes[2].set_ylabel('辐照度 (W/(m²·μm))', fontsize=11)
axes[2].set_title('太阳光谱与PDMS太阳吸收', fontsize=11)
axes[2].legend(fontsize=8)
axes[2].set_xlim(0.2, 4.0)

plt.suptitle(f'辐射制冷光谱分析 (d={d_detail:.0f}μm PDMS)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_spectral_analysis.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: fig_spectral_analysis.pdf")

# --- 图4: 各分量随厚度变化的堆叠图 ---
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.7
d_log_labels = [f'{d:.0f}' for d in d_arr]
x = np.arange(len(d_arr))

# 堆叠图
ax.bar(x, P_rad_arr, width, label='P_rad (向外辐射)', color='#e74c3c', alpha=0.8)
ax.bar(x, -P_atm_arr, width, label='P_atm (大气逆辐射)', color='#3498db', alpha=0.8)
ax.bar(x, -P_sun_arr, width, bottom=-P_atm_arr, label='P_sun (太阳吸收)', color='#f39c12', alpha=0.8)
# P_nonrad = 0 当 T=T_amb
net_bottom = -P_atm_arr - P_sun_arr
ax.bar(x, P_rad_arr, width, bottom=net_bottom,
       color='none', edgecolor='black', linewidth=1.5, hatch='//',
       label='P_rad - P_atm - P_sun')

# 标注净制冷功率
for i, d in enumerate(d_values):
    p_c = P_cool_arr[i]
    ax.annotate(f'{p_c:.0f}', (x[i], P_cool_arr[i] / 2),
                ha='center', va='center', fontsize=8, color='white',
                fontweight='bold' if p_c > 30 else 'normal')

ax.set_xticks(x)
ax.set_xticklabels(d_log_labels)
ax.set_xlabel('PDMS 膜厚 (μm)', fontsize=12)
ax.set_ylabel('功率 (W/m²)', fontsize=12)
ax.set_title('辐射制冷各分量随膜厚变化', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_power_components_bar.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: fig_power_components_bar.pdf")

# ============================================================
# 5. 保存结果
# ============================================================

results = {
    'environment': {
        'T_amb_K': T_amb,
        'T_amb_C': T_amb - 273.15,
        'h_c_W_m2K': h_c,
        'theta_sun_deg': np.degrees(theta_sun),
    },
    'best_cooling': {
        'd_opt_micron': best['d'],
        'P_cool_max_W_m2': best['P_cool'],
        'delta_T_K': best['delta_T'],
    },
    'all_results': [
        {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
         for k, v in p.items()}
        for p in performance
    ],
}

with open(os.path.join(RES_DIR, 'problem2_results.json'), 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# 保存性能表
with open(os.path.join(RES_DIR, 'cooling_performance_table.csv'), 'w') as f:
    headers = ['d_um', 'P_rad', 'P_atm', 'P_sun', 'P_nonrad', 'P_cool', 'delta_T', 'emis_8_13um', 'alpha_solar', 'eta']
    f.write(','.join(headers) + '\n')
    for p in performance:
        f.write(f"{p['d']:.1f},{p['P_rad']:.2f},{p['P_atm']:.2f},{p['P_sun']:.2f},{p['P_nonrad']:.2f},{p['P_cool']:.2f},{p['delta_T']:.3f},{p['emis_8_13um']:.4f},{p['alpha_solar']:.4f},{p['eta']:.4f}\n")

print("\n问题二完成! 结果已保存")
print(f"\n关键技术建议:")
print(f"  1. 推荐PDMS膜厚约为 {best['d']:.0f} μm, 此时净制冷功率 {best['P_cool']:.1f} W/m²")
print(f"  2. 当厚度超过 {best['d']:.0f} μm 后, 性能增益趋缓, 更多厚度不经济")
print(f"  3. 8-13μm发射率可达 {best_emis['emis_8_13um']:.4f}")
print(f"  4. 稳态温度降最大约 {best_dT['delta_T']:.1f} K")
print(f"  5. 需配合反射层减少太阳吸收, 太阳吸收率 {best['alpha_solar']:.4f}")
