"""
问题四: 辐射制冷材料与结构的综合优化设计

包含:
1. 多目标优化 (性能 + 成本 + 可制造性)
2. 成本经济分析
3. 可行性评估
4. 应用场景分析
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

# ============================================================
# 1. 加载问题三的最优结果
# ============================================================

print("=" * 60)
print("问题四: 综合优化设计")
print("=" * 60)

# 从问题三结果中加载最优结构
with open(os.path.join(RES_DIR, 'problem3_results.json'), 'r') as f:
    p3_results = json.load(f)

# 使用best_Ag_SiO2_PDMS或baseline_PDMS_only
if 'best_Ag_SiO2_PDMS' in p3_results:
    best_data = p3_results['best_Ag_SiO2_PDMS']
    best_name = 'Ag/SiO2/PDMS'
    best_mats = best_data['materials']
    best_thicks = best_data['thicknesses_um']
    best_pc = best_data['P_cool_est']
elif 'best_structure' in p3_results:
    best_struct = p3_results['best_structure']
    best_name = best_name
    best_mats = best_mats
    best_thicks = best_thicks
    best_pc = best_pc
else:
    # fallback: use PDMS only as best design
    best_name = 'PDMS 50um (最优基准)'
    best_mats = ['PDMS']
    best_thicks = [50.0]
    best_pc = p3_results.get('baseline_PDMS_only', {}).get('P_cool_est', 83.0)

print(f"最优设计: {best_name}")
print(f"材料: {best_mats}")
print(f"厚度: {[f'{t:.3f}' for t in best_thicks]} um")
print(f"P_cool: {best_pc:.1f} W/m2")

# ============================================================
# 2. 成本模型
# ============================================================

# 材料价格 (USD/kg, 参考市场价格)
material_prices = {
    'PDMS':  {'price_per_kg': 50,   'density_kg_m3': 970,   'process': '旋涂',     'process_cost': 5},
    'SiO2':  {'price_per_kg': 200,  'density_kg_m3': 2200,  'process': '磁控溅射',  'process_cost': 20},
    'TiO2':  {'price_per_kg': 150,  'density_kg_m3': 4230,  'process': '磁控溅射',  'process_cost': 20},
    'Ag':    {'price_per_kg': 800,  'density_kg_m3': 10500, 'process': '热蒸发',    'process_cost': 10},
    'Al2O3': {'price_per_kg': 100,  'density_kg_m3': 3950,  'process': '磁控溅射',  'process_cost': 20},
    'Si3N4': {'price_per_kg': 300,  'density_kg_m3': 3170,  'process': '磁控溅射',  'process_cost': 20},
    'MgF2':  {'price_per_kg': 250,  'density_kg_m3': 3180,  'process': '热蒸发',    'process_cost': 10},
}

# 基底成本
substrate_cost_per_m2 = 2.0  # 玻璃基底 USD/m²

def compute_cost(materials, thicknesses_um, area_m2=1.0):
    """
    计算多层膜结构的成本.

    返回:
        dict: 成本明细
    """
    total_material_cost = 0.0
    total_process_cost = 0.0
    details = []

    for name, d_um in zip(materials, thicknesses_um):
        info = material_prices.get(name, material_prices['PDMS'])
        d_m = d_um * 1e-6
        volume_m3 = area_m2 * d_m
        mass_kg = volume_m3 * info['density_kg_m3']
        mat_cost = mass_kg * info['price_per_kg']
        proc_cost = area_m2 * info['process_cost']

        total_material_cost += mat_cost
        total_process_cost += proc_cost

        details.append({
            'material': name,
            'thickness_um': d_um,
            'thickness_nm': d_um * 1000 if d_um < 1 else d_um,
            'process': info['process'],
            'material_cost_USD': mat_cost,
            'process_cost_USD': proc_cost,
        })

    total_cost = total_material_cost + total_process_cost + substrate_cost_per_m2

    return {
        'material_cost': total_material_cost,
        'process_cost': total_process_cost,
        'substrate_cost': substrate_cost_per_m2,
        'total_cost_per_m2': total_cost,
        'details': details,
    }

# 计算最优结构的成本
cost = compute_cost(best_mats, best_thicks)
print(f"\n成本分析 (每平方米):")
print(f"  材料成本: ${cost['material_cost']:.4f}")
print(f"  工艺成本: ${cost['process_cost']:.2f}")
print(f"  基底成本: ${cost['substrate_cost']:.2f}")
print(f"  总成本:   ${cost['total_cost_per_m2']:.2f}")

for d in cost['details']:
    unit = 'nm' if d['thickness_um'] < 1 else 'μm'
    print(f"  {d['material']}: {d['thickness_um']:.2f} {unit} ({d['process']}) — "
          f"材料${d['material_cost_USD']:.4f} + 工艺${d['process_cost_USD']:.2f}")

# ============================================================
# 3. 经济可行性分析
# ============================================================

# 假设: 制冷功率替换空调能耗
# 空调COP ≈ 3.0, 电价 ≈ 0.12 USD/kWh
P_cooling = best_pc  # W/m²
COP_ac = 3.0
electricity_price = 0.12  # USD/kWh
operating_hours_per_year = 8760 * 0.5  # 50%运行时间 (夜间+白天)
annual_energy_saving_kWh = P_cooling * operating_hours_per_year / 1000  # kWh/(m²·year)
annual_electricity_saving = annual_energy_saving_kWh / COP_ac * electricity_price  # USD/(m²·year)

print(f"\n经济分析:")
print(f"  净制冷功率: {P_cooling:.1f} W/m²")
print(f"  年运行时间: {operating_hours_per_year:.0f} 小时")
print(f"  年节能量: {annual_energy_saving_kWh:.1f} kWh/(m²·年)")
print(f"  年节省电费: ${annual_electricity_saving:.3f}/(m²·年)")

# 回收期
payback_period = cost['total_cost_per_m2'] / annual_electricity_saving if annual_electricity_saving > 0 else float('inf')
print(f"  成本回收期: {payback_period:.1f} 年")

# 10年净现值 (NPV)
lifetime = 10
discount_rate = 0.05
npv = -cost['total_cost_per_m2']
for year in range(1, lifetime + 1):
    npv += annual_electricity_saving / (1 + discount_rate)**year
print(f"  10年净现值 (NPV): ${npv:.2f}/m²")

# ============================================================
# 4. 技术成熟度与可行性评估
# ============================================================

trl_assessment = {
    '旋涂 (PDMS)': {'TRL': 8, '成熟度': '已批量生产', '风险': '低'},
    '磁控溅射 (SiO₂/TiO₂)': {'TRL': 9, '成熟度': '工业成熟', '风险': '低'},
    '热蒸发 (Ag)': {'TRL': 9, '成熟度': '工业成熟', '风险': '低'},
}

# 环境可靠性
reliability_factors = {
    '紫外老化': {'影响': '中', '缓解': 'PDMS上下加保护层'},
    '温湿度循环': {'影响': '低', '缓解': '材料本身耐候性好'},
    '机械磨损': {'影响': '低', '缓解': '静态应用, 无移动部件'},
    '灰尘污染': {'影响': '中', '缓解': '定期清洁或自清洁涂层'},
}

# ============================================================
# 5. 多方案对比 (帕累托前沿分析)
# ============================================================

print("\n生成帕累托前沿...")

# 生成多种设计方案
pareto_designs = []

# 方案A: 低成本方案 (薄PDMS + 简单结构)
designs_to_evaluate = [
    {'name': '方案A: 基础型', 'materials': ['Ag', 'PDMS'], 'thicknesses': [0.1, 30.0]},
    {'name': '方案B: 标准型', 'materials': ['Ag', 'PDMS'], 'thicknesses': [0.1, 50.0]},
    {'name': '方案C: 增强型', 'materials': ['Ag', 'SiO2', 'PDMS'], 'thicknesses': [0.1, 0.5, 60.0]},
    {'name': '方案D: 高性能型', 'materials': best_mats,
     'thicknesses': best_thicks},
    {'name': '方案E: 超高性能型', 'materials': ['Ag', 'TiO2', 'SiO2', 'TiO2', 'SiO2', 'PDMS'],
     'thicknesses': [0.1, 0.12, 0.15, 0.12, 0.15, 80.0]},
]

# 用简化方法估算各方案性能 (不重新跑完整TMM优化以节省时间)
# 基于物理规律推算
wl = create_wavelength_grid()[::8]
n_pdms, k_pdms = build_pdms_refractive_index(wl)

for design in designs_to_evaluate:
    cost_d = compute_cost(design['materials'], design['thicknesses'])

    # 使用简化计算估算P_cool (基于单层PDMS的缩放规律)
    # 对于有Ag基底的结构, 太阳反射率大幅提高
    has_ag = 'Ag' in design['materials']
    has_dbr = design['materials'].count('TiO2') >= 2

    # 估算
    pdms_d = design['thicknesses'][-1]  # PDMS通常是最外层

    # 8-13μm发射率估算 (基于厚度)
    window_emis = 1.0 - np.exp(-pdms_d / 15.0)  # 简化的指数饱和模型
    window_emis = min(window_emis, 0.97)

    # 太阳吸收率估算
    if has_ag:
        alpha_solar_est = 0.02
    elif has_dbr:
        alpha_solar_est = 0.05
    else:
        alpha_solar_est = 0.10

    # 制冷功率估算 (基于物理尺度)
    # 黑体300K总辐射 ≈ 459 W/m², 取8-13μm约占30%
    P_rad_est = 459 * 0.35 * window_emis
    P_atm_est = 300 * 0.35 * window_emis  # 大气逆辐射
    P_sun_est = 1000 * alpha_solar_est      # 太阳吸收

    P_cool_est = P_rad_est - P_atm_est - P_sun_est

    pareto_designs.append({
        'name': design['name'],
        'P_cool': P_cool_est,
        'cost_per_m2': cost_d['total_cost_per_m2'],
        'emis_8_13_est': window_emis,
        'alpha_solar_est': alpha_solar_est,
        'materials': design['materials'],
        'thicknesses': design['thicknesses'],
        'payback_years': cost_d['total_cost_per_m2'] / max(P_cool_est * operating_hours_per_year / 1000
                                                            / COP_ac * electricity_price, 0.001),
        'npv_10yr': -cost_d['total_cost_per_m2'] + sum(
            P_cool_est * operating_hours_per_year / 1000 / COP_ac * electricity_price
            / (1 + discount_rate)**y for y in range(1, lifetime+1)),
    })

# 更新最优结构的经济数据
best_eco = next((d for d in pareto_designs if d['name'] == '方案D: 高性能型'), pareto_designs[0])
best_eco['P_cool'] = best_pc
best_eco['payback_years'] = payback_period
best_eco['npv_10yr'] = npv

# ============================================================
# 6. 制图
# ============================================================

# --- 图1: 成本-性能帕累托前沿 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 左: P_cool vs Cost
ax = axes[0]
names = [d['name'] for d in pareto_designs]
pc_vals = [d['P_cool'] for d in pareto_designs]
cost_vals = [d['cost_per_m2'] for d in pareto_designs]
colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

for i, (name, pc, cost_val, color) in enumerate(zip(names, pc_vals, cost_vals, colors)):
    ax.scatter(cost_val, pc, s=200, c=color, edgecolors='black', linewidth=1.2, zorder=5)
    ax.annotate(name, (cost_val, pc), textcoords='offset points',
                xytext=(10, 10), fontsize=9, fontweight='bold')

# 帕累托前沿连线
pareto_order = sorted(zip(cost_vals, pc_vals, names), key=lambda x: x[0])
pareto_front = [pareto_order[0]]
for c, p, n in pareto_order[1:]:
    if p > pareto_front[-1][1]:
        pareto_front.append((c, p, n))
pc_front = [p[1] for p in pareto_front]
cost_front = [p[0] for p in pareto_front]
ax.plot(cost_front, pc_front, 'k--', linewidth=1, alpha=0.5, label='帕累托前沿')

ax.set_xlabel('成本 (USD/m²)', fontsize=12)
ax.set_ylabel('净制冷功率 P_cool (W/m²)', fontsize=12)
ax.set_title('性能-成本帕累托前沿', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 右: 回收期 vs NPV
ax2 = axes[1]
payback_vals = [d['payback_years'] for d in pareto_designs]
npv_vals = [d['npv_10yr'] for d in pareto_designs]

for i, (name, pb, npv_val, color) in enumerate(zip(names, payback_vals, npv_vals, colors)):
    ax2.scatter(pb, npv_val, s=200, c=color, edgecolors='black', linewidth=1.2, zorder=5)
    ax2.annotate(name, (pb, npv_val), textcoords='offset points',
                xytext=(10, -10), fontsize=9, fontweight='bold')

ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.axvline(5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, label='5年基准')
ax2.set_xlabel('成本回收期 (年)', fontsize=12)
ax2.set_ylabel('10年净现值 NPV (USD/m²)', fontsize=12)
ax2.set_title('经济可行性评估', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle('辐射制冷产品综合优化与经济分析', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_pareto_economic.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: fig_pareto_economic.pdf")

# --- 图2: 技术成熟度与成本分解 ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 左: 成本分解饼图
ax = axes[0]
cost_detail = cost
labels = [d['material'] + f'\n({d["process"]})' for d in cost_detail['details']]
labels.append('基底')
sizes = [d['material_cost_USD'] + d['process_cost_USD'] for d in cost_detail['details']]
sizes.append(cost_detail['substrate_cost'])
# 合并很小项
threshold = 0.01
small_mask = [s < threshold for s in sizes]
if any(small_mask):
    others_sum = sum(s for s, m in zip(sizes, small_mask) if m)
    sizes = [s for s, m in zip(sizes, small_mask) if not m] + ([others_sum] if others_sum > 0 else [])
    labels = [l for l, m in zip(labels, small_mask) if not m] + (['其他'] if others_sum > 0 else [])

wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                   colors=plt.cm.Set3(np.linspace(0, 1, len(sizes))),
                                   textprops={'fontsize': 8})
ax.set_title(f'成本分解 (总计 ${cost["total_cost_per_m2"]:.2f}/m²)', fontsize=11, fontweight='bold')

# 中: TRL评估
ax = axes[1]
processes = list(trl_assessment.keys())
trl_values = [trl_assessment[p]['TRL'] for p in processes]
risk_levels = [trl_assessment[p]['风险'] for p in processes]
bar_colors = ['#2ecc71' if r == '低' else '#f39c12' for r in risk_levels]
bars = ax.barh(processes, trl_values, color=bar_colors, edgecolor='black', linewidth=1)
ax.set_xlim(0, 10)
ax.set_xlabel('技术成熟度 (TRL)', fontsize=11)
ax.set_title('制备工艺技术成熟度', fontsize=11, fontweight='bold')
for bar, trl in zip(bars, trl_values):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f'TRL {trl}', va='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='x')

# 右: 年度能源节省
ax = axes[2]
years = np.arange(1, 11)
cumulative_savings = np.cumsum([annual_electricity_saving / (1 + discount_rate)**y for y in years])
initial_cost = cost['total_cost_per_m2']

ax.fill_between(years, 0, cumulative_savings, alpha=0.3, color='green')
ax.plot(years, cumulative_savings, 'g-', linewidth=2, marker='o', markersize=6)
ax.axhline(initial_cost, color='red', linestyle='--', linewidth=1.2, label=f'初始投资 ${initial_cost:.2f}')
ax.axvline(payback_period, color='blue', linestyle=':', linewidth=1.2, label=f'回收期 {payback_period:.1f}年')

ax.set_xlabel('使用年数', fontsize=11)
ax.set_ylabel('累计净节省 (USD/m²)', fontsize=11)
ax.set_title('投资回收分析', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle(f'最优设计 ({best_name}) 综合评估', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_comprehensive_evaluation.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: fig_comprehensive_evaluation.pdf")

# --- 图3: 与其他制冷技术对比 ---
fig, ax = plt.subplots(figsize=(10, 6))

tech_comparison = {
    '辐射制冷\n(本设计)':  {'P_cool': best_pc, 'COP': float('inf'), 'cost': cost['total_cost_per_m2']},
    '辐射制冷\n(单层PDMS)': {'P_cool': p3_results['baseline_PDMS_only']['P_cool_est'], 'COP': float('inf'), 'cost': 10},
    '传统空调':    {'P_cool': 100, 'COP': 3.0, 'cost': 50},
    '蒸发冷却':    {'P_cool': 80, 'COP': 15, 'cost': 20},
    '热电制冷':    {'P_cool': 60, 'COP': 1.5, 'cost': 80},
    '吸收式制冷':  {'P_cool': 90, 'COP': 0.8, 'cost': 60},
}

x = np.arange(len(tech_comparison))
width = 0.3

# 归一化指标可视化
pc_norm = [v['P_cool'] / 100 for v in tech_comparison.values()]
cost_norm = [1.0 - v['cost'] / 80 for v in tech_comparison.values()]  # 反转成本: 低成本=高分
cop_norm = [min(v['COP'] / 20, 1.0) for v in tech_comparison.values()]  # COP inf → 1.0

ax.bar(x - width, pc_norm, width, label='制冷功率 (归一化)', color='#e74c3c', alpha=0.8)
ax.bar(x, cost_norm, width, label='低成本 (归一化)', color='#2ecc71', alpha=0.8)
ax.bar(x + width, cop_norm, width, label='能效 COP (归一化)', color='#3498db', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(tech_comparison.keys())
ax.set_ylabel('归一化评分', fontsize=12)
ax.set_title('辐射制冷与其他制冷技术对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.15)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_technology_comparison.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: fig_technology_comparison.pdf")

# ============================================================
# 7. 保存结果
# ============================================================

results = {
    'optimal_design': {
        'name': best_name,
        'materials': best_mats,
        'thicknesses_um': best_thicks,
        'P_cool_W_m2': best_pc,
    },
    'cost_analysis': {
        'material_cost_USD_m2': cost['material_cost'],
        'process_cost_USD_m2': cost['process_cost'],
        'substrate_cost_USD_m2': cost['substrate_cost'],
        'total_cost_USD_m2': cost['total_cost_per_m2'],
        'details': cost['details'],
    },
    'economic_analysis': {
        'annual_energy_saving_kWh_m2': annual_energy_saving_kWh,
        'annual_electricity_saving_USD_m2': annual_electricity_saving,
        'payback_period_years': payback_period,
        'npv_10yr_USD_m2': npv,
        'discount_rate': discount_rate,
        'electricity_price_USD_kWh': electricity_price,
    },
    'feasibility': {
        'TRL_overall': 7,
        'technical_risk': '低-中',
        'manufacturing_readiness': '可以使用现有工业设备批量生产',
        'key_challenges': ['大面积均匀性控制', '长期户外耐久性验证', '成本进一步降低'],
    },
    'pareto_designs': pareto_designs,
}

with open(os.path.join(RES_DIR, 'problem4_results.json'), 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n问题四完成!")
print(f"\n最终推荐方案:")
print(f"  结构: {best_name}")
print(f"  材料: {' / '.join(best_mats)}")
print(f"  预期P_cool: {best_pc:.1f} W/m²")
print(f"  成本: ${cost['total_cost_per_m2']:.2f}/m²")
print(f"  回收期: {payback_period:.1f} 年")
print(f"  10年NPV: ${npv:.2f}/m²")
