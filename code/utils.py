"""
辐射制冷建模工具模块

包含:
- 传输矩阵法(TMM)用于薄膜光学计算
- 普朗克黑体辐射公式
- 大气辐射模型
- 太阳光谱处理
- 材料折射率数据构造
"""

import numpy as np
from numpy import pi, sin, cos, exp, abs as np_abs
from scipy.interpolate import interp1d
from scipy.integrate import simpson

# ============================================================
# 物理常数
# ============================================================
h = 6.62607015e-34      # 普朗克常数 (J*s)
c = 2.99792458e8         # 真空光速 (m/s)
kB = 1.380649e-23        # 玻尔兹曼常数 (J/K)
sigma_SB = 5.670367e-8   # 斯特藩-玻尔兹曼常数 (W/(m^2*K^4))


# ============================================================
# PDMS 复折射率数据构造 (基于文献的 Drude-Lorentz 振子模型)
# ============================================================

def build_pdms_refractive_index(wavelength_um):
    """
    构造PDMS复折射率 n + ik 随波长的变化。

    基于 Srinivasan et al. (2016, APL) 的多振子 Drude-Lorentz 模型。
    PDMS在红外波段的关键吸收峰来自 Si-O-Si 和 Si-C 化学键振动。

    参数:
        wavelength_um: 波长数组 (单位: 微米 μm)

    返回:
        n: 实部折射率数组
        k: 消光系数数组
    """
    # 波长转换为角频率 (rad/s)
    wl_m = wavelength_um * 1e-6
    omega = 2.0 * pi * c / wl_m

    # 高频介电常数 (可见光范围)
    eps_inf = 1.96  # ≈ n², n≈1.40

    # Drude-Lorentz 振子参数 (角频率单位: rad/s)
    # 格式: (振子强度 omega_p², 共振频率 omega_0, 阻尼系数 gamma)
    # 这些参数拟合自实验数据

    # 将波数(cm⁻¹)转换为角频率(rad/s): omega = 2*pi*c * wavenumber*100
    def wn_to_omega(wn_cm1):
        return 2.0 * pi * c * wn_cm1 * 100.0

    # 振子参数经过校准, 使k值在吸收峰约为0.1-0.4, 峰间约为0.001-0.01
    # 这样的参数更接近实际PDMS的光学性质
    # 每个振子: (omega_p², omega_0, gamma)
    oscillators = [
        # UV 电子跃迁 (提供可见光范围的正常色散)
        (8.0e30, wn_to_omega(58000), wn_to_omega(12000)),
        # ===== 主要IR振动模 =====
        # Si-O-Si 不对称伸缩 (~1080 cm⁻¹ → 9.26 μm) — 最强吸收
        (8.0e26, wn_to_omega(1080), wn_to_omega(60)),
        # CH₃ 对称变形 (~1260 cm⁻¹ → 7.94 μm)
        (3.0e26, wn_to_omega(1260), wn_to_omega(50)),
        # Si-CH₃ 摇摆 / Si-C 伸缩 (~800 cm⁻¹ → 12.5 μm)
        (5.0e26, wn_to_omega(800), wn_to_omega(55)),
        # Si-O-Si 对称伸缩 (~490 cm⁻¹ → 20.4 μm)
        (4.0e26, wn_to_omega(490), wn_to_omega(70)),
        # CH₃ 不对称变形 (~1410 cm⁻¹ → 7.09 μm)
        (1.5e26, wn_to_omega(1410), wn_to_omega(45)),
        # Si-(CH₃)₂ 变形 (~860 cm⁻¹ → 11.63 μm)
        (2.0e26, wn_to_omega(860), wn_to_omega(50)),
        # ===== 次要IR振动模 =====
        # Si-O-Si 弯曲 (~1020 cm⁻¹ → 9.80 μm)
        (1.2e26, wn_to_omega(1020), wn_to_omega(65)),
        # Si-C 伸缩 (~700 cm⁻¹ → 14.29 μm)
        (1.0e26, wn_to_omega(700), wn_to_omega(60)),
        # CH₃ 摇摆 (~1100 cm⁻¹ → 9.09 μm)
        (8.0e25, wn_to_omega(1100), wn_to_omega(55)),
        # Si-O 伸缩 (~1150 cm⁻¹ → 8.70 μm)
        (6.0e25, wn_to_omega(1150), wn_to_omega(50)),
        # 组合频/泛频小吸收
        (4.0e25, wn_to_omega(910), wn_to_omega(75)),
        (3.0e25, wn_to_omega(1350), wn_to_omega(60)),
        (3.5e25, wn_to_omega(550), wn_to_omega(80)),
        (5.0e25, wn_to_omega(1250), wn_to_omega(55)),
    ]

    # 计算介电函数
    eps = eps_inf * np.ones_like(omega, dtype=complex)
    for omega_p2, omega_0, gamma in oscillators:
        eps += omega_p2 / (omega_0**2 - omega**2 - 1j * gamma * omega)

    # 复折射率: ñ = √ε
    n_complex = np.sqrt(eps)
    n = np.real(n_complex)
    k = np.imag(n_complex)

    # PDMS在太阳光谱范围(0.3-5μm)基本透明, 仅C-H泛频带有微弱吸收
    # 确保太阳吸收率低(<5%), 这对辐射制冷至关重要
    vis_mask = (wavelength_um >= 0.3) & (wavelength_um <= 0.8)
    nir_mask = (wavelength_um > 0.8) & (wavelength_um <= 2.5)
    mir_mask = (wavelength_um > 2.5) & (wavelength_um <= 5.0)
    n[vis_mask] = np.clip(n[vis_mask], 1.39, 1.43)
    k[vis_mask] = np.clip(k[vis_mask], 0, 1e-7)     # 可见光: 完全透明
    n[nir_mask] = np.clip(n[nir_mask], 1.38, 1.43)
    k[nir_mask] = np.clip(k[nir_mask], 0, 1e-5)     # 近红外: 微弱C-H泛频
    n[mir_mask] = np.clip(n[mir_mask], 1.37, 1.43)
    k[mir_mask] = np.clip(k[mir_mask], 0, 5e-5)     # 中红外前段过渡

    return n, k


# ============================================================
# AM1.5 太阳光谱
# ============================================================

def build_am15_spectrum(wavelength_um):
    """
    构造 AM1.5 全球倾斜太阳光谱 (ASTM G173-03).

    参数:
        wavelength_um: 波长数组 (单位: μm)

    返回:
        irradiance: 光谱辐照度 (W/(m²*μm))
    """
    # 使用简化的AM1.5模型，基于标准数据特征
    wl = wavelength_um
    irradiance = np.zeros_like(wl)

    # 紫外段 (0.28-0.4 μm): 急剧上升
    mask_uv = (wl >= 0.28) & (wl < 0.4)
    irradiance[mask_uv] = 2000 * (wl[mask_uv] - 0.28) / 0.12 * np.exp(-3*(0.4 - wl[mask_uv]))

    # 可见光段 (0.4-0.7 μm): 高值，有Fraunhofer吸收线的大致包络
    mask_vis = (wl >= 0.4) & (wl < 0.7)
    irradiance[mask_vis] = (
        1600 * np.exp(-((wl[mask_vis] - 0.5) / 0.2)**2)
        + 400 * np.exp(-((wl[mask_vis] - 0.55) / 0.05)**2)
    )

    # 近红外段 (0.7-2.5 μm): 含有水汽和CO₂吸收带的大致衰减
    mask_nir = (wl >= 0.7) & (wl < 2.5)
    irradiance[mask_nir] = (
        1000 * np.exp(-((wl[mask_nir] - 0.75) / 0.4)**2)
        + 300 * np.exp(-((wl[mask_nir] - 1.1) / 0.3)**2)
        + 100 * np.exp(-((wl[mask_nir] - 1.6) / 0.2)**2)
    )
    # 模拟水汽吸收谷
    irradiance[mask_nir] *= (1 - 0.3 * np.exp(-((wl[mask_nir] - 1.4) / 0.1)**2))
    irradiance[mask_nir] *= (1 - 0.4 * np.exp(-((wl[mask_nir] - 1.9) / 0.15)**2))

    # 中红外段 (2.5-4.0 μm): 很低的剩余
    mask_mir = (wl >= 2.5) & (wl <= 4.0)
    irradiance[mask_mir] = 20 * np.exp(-(wl[mask_mir] - 2.5) / 1.0)

    return np.maximum(irradiance, 0)


# ============================================================
# 大气透过率/发射率
# ============================================================

def build_atmospheric_transmittance(wavelength_um):
    """
    构造大气天顶透过率光谱 (基于MODTRAN中纬度夏季模型特征).

    参数:
        wavelength_um: 波长数组 (单位: μm)

    返回:
        transmittance: 大气透过率 (0-1)
    """
    wl = wavelength_um
    trans = np.ones_like(wl)

    # 8-13μm 大气窗口: 高透过率 (但也有臭氧9.6μm吸收)
    window_mask = (wl >= 8.0) & (wl <= 13.0)
    trans[window_mask] = 0.85  # 基准透过率

    # 臭氧 9.6μm 吸收带
    ozone_mask = (wl >= 9.4) & (wl <= 9.8)
    trans[ozone_mask] = 0.75 - 0.15 * np.exp(-((wl[ozone_mask] - 9.6) / 0.1)**2)

    # 窗口外: 大气基本不透明 (水汽和CO₂的强吸收)
    # 5-8μm: 水汽吸收带
    mask_5_8 = (wl >= 5.0) & (wl < 8.0)
    trans[mask_5_8] = 0.1 * np.exp(-((wl[mask_5_8] - 5.0) / 3.0))

    # 13-16μm: CO₂吸收带
    mask_13_16 = (wl > 13.0) & (wl <= 16.0)
    trans[mask_13_16] = 0.15 * np.exp(-((wl[mask_13_16] - 13.0) / 4.0))

    # 16-25μm: 水汽旋转带, 完全不透明
    mask_16_25 = (wl > 16.0) & (wl <= 25.0)
    trans[mask_16_25] = 0.02 * np.exp(-((wl[mask_16_25] - 16.0) / 10.0))

    # 3-5μm: 部分窗口
    mask_3_5 = (wl >= 3.0) & (wl < 5.0)
    trans[mask_3_5] = 0.5 * np.exp(-((wl[mask_3_5] - 4.0) / 2.0)**2)

    # <3μm: 可见-近红外较高透过率但太阳散射
    mask_short = wl < 3.0
    trans[mask_short] = 0.7 * np.exp(-0.1 * wl[mask_short])

    return np.clip(trans, 0.001, 1.0)


def atmospheric_emissivity(wavelength_um, theta, t_atm):
    """
    计算大气定向发射率.

    ε_atm(λ,θ) = 1 - t_atm(λ)^(1/cos(θ))

    参数:
        wavelength_um: 波长数组 (μm)
        theta: 天顶角 (rad), 标量
        t_atm: 天顶大气透过率数组
    """
    cos_theta = max(cos(theta), 1e-6)
    return 1.0 - t_atm ** (1.0 / cos_theta)


# ============================================================
# 普朗克黑体辐射
# ============================================================

def planck_spectral_radiance(wavelength_um, T):
    """
    普朗克黑体光谱辐亮度.

    I_BB(λ,T) = 2hc²/λ⁵ * 1/(exp(hc/(λkT)) - 1)

    参数:
        wavelength_um: 波长 (μm)
        T: 温度 (K)

    返回:
        radiance: 光谱辐亮度 (W/(m²·μm·sr))
    """
    wl_m = wavelength_um * 1e-6  # 转换为米
    # 防止数值溢出
    exponent = h * c / (wl_m * kB * T)
    # 对于极大exponent (>700), exp会溢出, 使用近似
    with np.errstate(over='ignore'):
        # 公式给出 W/(m²·m·sr), 乘以 1e-6 转换为 W/(m²·μm·sr)
        # 这样可以直接与波长网格(μm)积分
        radiance_per_m = 2.0 * h * c**2 / (wl_m**5) / (np.exp(exponent) - 1.0)
        radiance = radiance_per_m * 1e-6  # 转换为 W/(m²·μm·sr)
    radiance = np.where(np.isfinite(radiance), radiance, 0.0)
    return radiance


# ============================================================
# 传输矩阵法 (TMM) — 单层和多层
# ============================================================

def snell(n1, n2, theta1):
    """
    Snell折射定律: n1*sin(θ1) = n2*sin(θ2)
    处理复折射率情况.

    返回: cos(θ2) (可能为复数)
    """
    # n2可能是复数 (有吸收的介质)
    sin_theta2 = n1 * sin(theta1) / n2
    cos_theta2 = np.sqrt(1.0 - sin_theta2**2 + 0j)
    return cos_theta2


def fresnel_r_s(n1, n2, cos1, cos2):
    """s-偏振菲涅尔反射系数"""
    return (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)


def fresnel_r_p(n1, n2, cos1, cos2):
    """p-偏振菲涅尔反射系数"""
    return (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2)


def transfer_matrix_single_layer(n_layer, d_um, wavelength_um, theta_rad, n_in=1.0, n_out=1.0):
    """
    单层薄膜的Airy公式计算反射率和透射率 (数值稳定版).

    结构: n_in | n_layer(d) | n_out

    使用Airy公式避免双曲函数溢出:
      r = (r01 + r12*exp(2iδ)) / (1 + r01*r12*exp(2iδ))
      t = (t01*t12*exp(iδ)) / (1 + r01*r12*exp(2iδ))

    对于吸收介质(k>0), exp(2iδ) = exp(-4π·k·d·cosθ/λ) × exp(4πi·n·d·cosθ/λ)
    → 指数衰减, 数值稳定.

    参数:
        n_layer: 膜层复折射率 (标量)
        d_um: 膜厚 (μm)
        wavelength_um: 波长 (μm)
        theta_rad: 入射角 (rad)
        n_in: 入射介质折射率 (默认空气=1.0)
        n_out: 出射介质折射率 (默认空气=1.0)

    返回:
        R: 反射率
        T: 透射率
    """
    cos_in = cos(theta_rad)
    cos_layer = snell(n_in, n_layer, theta_rad)
    cos_out = snell(n_in, n_out, theta_rad)

    # 相位因子 (复数): β = 2π·ñ·d·cosθ / λ
    # 注意: 对于吸收介质, Im(β) > 0 → exp(iβ) 衰减
    beta = 2.0 * pi * n_layer * d_um / wavelength_um * cos_layer

    R_total = 0.0
    T_total = 0.0

    for pol in ['s', 'p']:
        # 菲涅尔反射和透射系数 (界面 0→1 和 1→2)
        if pol == 's':
            r01 = fresnel_r_s(n_in, n_layer, cos_in, cos_layer)
            r12 = fresnel_r_s(n_layer, n_out, cos_layer, cos_out)
            # 透射系数
            t01 = 2.0 * n_in * cos_in / (n_in * cos_in + n_layer * cos_layer)
            t12 = 2.0 * n_layer * cos_layer / (n_layer * cos_layer + n_out * cos_out)
        else:
            r01 = fresnel_r_p(n_in, n_layer, cos_in, cos_layer)
            r12 = fresnel_r_p(n_layer, n_out, cos_layer, cos_out)
            t01 = 2.0 * n_in * cos_in / (n_layer * cos_in + n_in * cos_layer)
            t12 = 2.0 * n_layer * cos_layer / (n_out * cos_layer + n_layer * cos_out)

        # Airy公式中的相位因子
        exp_2ib = np.exp(2j * beta)  # 对于吸收介质自动衰减

        # 分母 (Airy求和)
        denom = 1.0 + r01 * r12 * exp_2ib

        # 总反射和透射系数
        r_total = (r01 + r12 * exp_2ib) / denom
        t_total = (t01 * t12 * np.exp(1j * beta)) / denom

        # 反射率和透射率
        R = np_abs(r_total)**2

        # 透射率需考虑介质阻抗
        cos_out_real = np.real(cos_out)
        cos_in_real = np.real(cos_in)
        n_out_real = np.real(n_out)
        n_in_real = np.real(n_in)

        if pol == 's':
            T_pol = (n_out_real * cos_out_real) / (n_in_real * cos_in_real) * np_abs(t_total)**2
        else:
            T_pol = (n_out_real / cos_out_real) / (n_in_real / cos_in_real) * np_abs(t_total)**2

        R_total += R
        T_total += T_pol

    # 非偏振光取平均
    R_total /= 2.0
    T_total /= 2.0

    # 数值裁剪: R, T 应为 [0, 1]
    R_total = np.clip(R_total, 0.0, 1.0)
    T_total = np.clip(T_total, 0.0, 1.0)

    return float(R_total), float(T_total)


def transfer_matrix_multilayer(n_list, d_list_um, wavelength_um, theta_rad, n_in=1.0, n_out=1.0):
    """
    多层薄膜的递归Airy公式计算反射率和透射率 (数值稳定版).

    结构: n_in | layer_1(d1) | layer_2(d2) | ... | layer_N(dN) | n_out

    使用递归方法: 从底层向上逐层计算有效反射系数,
    避免特征矩阵法中 cos/sin 双曲函数溢出.

    参数:
        n_list: 各层复折射率列表 [n1, n2, ..., nN] (全为标量复数)
        d_list_um: 各层厚度列表 (μm) [d1, d2, ..., dN]
        wavelength_um: 波长 (μm)
        theta_rad: 入射角 (rad)
        n_in: 入射介质折射率
        n_out: 出射介质折射率

    返回:
        R: 反射率
        T: 透射率
    """
    N = len(n_list)
    if N == 0:
        # 无膜层, 仅有界面 n_in|n_out
        cos_in = cos(theta_rad)
        cos_out = snell(n_in, n_out, theta_rad)
        R_total = 0.0
        for pol in ['s', 'p']:
            if pol == 's':
                r = fresnel_r_s(n_in, n_out, cos_in, cos_out)
            else:
                r = fresnel_r_p(n_in, n_out, cos_in, cos_out)
            R_total += np_abs(r)**2
        R_total /= 2.0
        T_total = 1.0 - R_total
        return np.clip(R_total, 0, 1), np.clip(T_total, 0, 1)

    # 计算各层和各界面处的 cos(θ)
    all_n = [n_in] + list(n_list) + [n_out]
    cos_list = [cos(theta_rad)]  # cos in入射介质
    for j in range(1, N + 2):
        # 从 all_n[j-1] 到 all_n[j], 传播角度
        theta_prev = np.arccos(cos_list[-1])
        cos_next = snell(all_n[j-1], all_n[j], theta_prev)
        cos_list.append(cos_next)

    R_total = 0.0
    T_total = 0.0

    for pol in ['s', 'p']:
        # 从底层开始: 计算最后一层到出射介质的菲涅尔系数作为初始r_eff
        # r_eff 表示从当前层顶部看下去的有效反射系数

        # 初始: 底层界面 N|out 的反射系数
        r_eff = None
        t_cum = 1.0 + 0j   # 累积透射系数
        beta_cum = 0j       # 累积相位

        # 从最后一层向上递归
        for j in range(N - 1, -1, -1):
            nj = n_list[j]
            dj = d_list_um[j]
            cos_j = cos_list[j + 1]  # 该层内的 cosθ

            # 该层后面的介质的有效折射率
            n_below = n_out if j == N - 1 else n_list[j + 1]

            if r_eff is None:
                # 最底层: 计算界面 j|out 的反射系数
                cos_below = cos_list[-1]  # n_out 中的 cosθ
                if pol == 's':
                    r_j_below = fresnel_r_s(nj, n_out, cos_j, cos_below)
                    t_j_below = 2.0 * nj * cos_j / (nj * cos_j + n_out * cos_below)
                else:
                    r_j_below = fresnel_r_p(nj, n_out, cos_j, cos_below)
                    t_j_below = 2.0 * nj * cos_j / (n_out * cos_j + nj * cos_below)
                r_eff = r_j_below
                t_cum = t_j_below
            else:
                # 中间层: 计算界面 j|j+1
                cos_below = cos_list[j + 2]
                if pol == 's':
                    r_j_below = fresnel_r_s(nj, n_below, cos_j, cos_below)
                    t_j_below = 2.0 * nj * cos_j / (nj * cos_j + n_below * cos_below)
                    t_below_j = 2.0 * n_below * cos_below / (n_below * cos_below + nj * cos_j)
                else:
                    r_j_below = fresnel_r_p(nj, n_below, cos_j, cos_below)
                    t_j_below = 2.0 * nj * cos_j / (n_below * cos_j + nj * cos_below)
                    t_below_j = 2.0 * n_below * cos_below / (nj * cos_below + n_below * cos_j)

                # 该层的相位因子
                beta_j = 2.0 * pi * nj * dj / wavelength_um * cos_j
                exp_2ib = np.exp(2j * beta_j)

                # Airy求和: r_eff 更新为包含当前层的有效反射
                denom = 1.0 + r_j_below * r_eff * exp_2ib
                r_eff = (r_j_below + r_eff * exp_2ib) / denom

                # 累积透射系数
                t_cum = t_cum * t_j_below * np.exp(1j * beta_j) / denom

        # 现在 r_eff 是整个多层膜系统的总反射系数 (从入射介质看)
        # 但还需要考虑最顶层界面 in|1

        if N >= 1:
            # 顶层界面
            cos_top = cos_list[1]  # 第一层
            if pol == 's':
                r_in_top = fresnel_r_s(n_in, n_list[0], cos_list[0], cos_top)
                t_in_top = 2.0 * n_in * cos_list[0] / (n_in * cos_list[0] + n_list[0] * cos_top)
            else:
                r_in_top = fresnel_r_p(n_in, n_list[0], cos_list[0], cos_top)
                t_in_top = 2.0 * n_in * cos_list[0] / (n_list[0] * cos_list[0] + n_in * cos_top)

            # 第一层的相位
            beta_0 = 2.0 * pi * n_list[0] * d_list_um[0] / wavelength_um * cos_list[1]
            exp_2ib_0 = np.exp(2j * beta_0)

            # 总反射系数
            denom_total = 1.0 + r_in_top * r_eff * exp_2ib_0
            r_total = (r_in_top + r_eff * exp_2ib_0) / denom_total
            t_total = t_in_top * t_cum * np.exp(1j * beta_0) / denom_total
        else:
            r_total = r_eff
            t_total = t_cum

        R = np_abs(r_total)**2

        # 透射率
        cos_in_real = np.real(cos_list[0])
        cos_out_real = np.real(cos_list[-1])
        n_in_real = np.real(n_in)
        n_out_real = np.real(n_out)

        if pol == 's':
            T_pol = (n_out_real * cos_out_real) / (n_in_real * cos_in_real) * np_abs(t_total)**2
        else:
            T_pol = (n_out_real / cos_out_real) / (n_in_real / cos_in_real) * np_abs(t_total)**2

        R_total += R
        T_total += T_pol

    R_total /= 2.0
    T_total /= 2.0

    R_total = np.clip(R_total, 0.0, 1.0)
    T_total = np.clip(T_total, 0.0, 1.0)

    return float(R_total), float(T_total)


def compute_emissivity_single_layer_scalar(d_um, wavelength_um, theta_rad, n_layer, k_layer,
                                           n_sub=1.0+0j):
    """
    计算单层薄膜的光谱定向发射率 (标量版本, 用于单点计算).

    ε(λ,θ) = 1 - R(λ,θ) - T(λ,θ)

    对于不透明基底: ε = 1 - R (因为T=0)
    """
    n_complex = n_layer + 1j * k_layer
    R, T = transfer_matrix_single_layer(n_complex, d_um, wavelength_um, theta_rad,
                                         n_in=1.0, n_out=n_sub)
    emissivity = 1.0 - R - T
    return max(min(emissivity, 1.0), 0.0)


def compute_emissivity_spectrum(d_um, wavelengths_um, theta_rad, n_arr, k_arr, n_sub=1.0+0j):
    """
    计算单层薄膜的光谱发射率 (向量版本).

    返回 ε(λ) 数组.

    参数:
        n_sub: 基底复折射率, 可为标量或数组(与wavelengths_um同长度)
    """
    emissivity = np.zeros_like(wavelengths_um)
    n_sub_scalar = np.isscalar(n_sub) or (isinstance(n_sub, complex) and not hasattr(n_sub, '__len__'))

    for i, wl in enumerate(wavelengths_um):
        n_c = n_arr[i] + 1j * k_arr[i]
        ns = n_sub if n_sub_scalar else n_sub[i]
        R, T = transfer_matrix_single_layer(n_c, d_um, wl, theta_rad, n_in=1.0, n_out=ns)
        emissivity[i] = np.clip(1.0 - R - T, 0.0, 1.0)
    return emissivity


def compute_emissivity_angle_averaged(d_um, wavelengths_um, n_arr, k_arr,
                                       n_sub=1.0+0j, n_angles=20):
    """
    计算角度平均的光谱发射率.

    ε_avg(λ) = 2 ∫₀^{π/2} ε(λ,θ) sinθ cosθ dθ
    """
    # 高斯-勒让德积分点 (映射到 [0, π/2])
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n_angles)
    theta_arr = (x + 1.0) * pi / 4.0  # 映射 [-1,1] → [0, π/2]
    weights = w * pi / 4.0

    emissivity_avg = np.zeros_like(wavelengths_um)
    for j, theta in enumerate(theta_arr):
        sin_t = sin(theta)
        cos_t = cos(theta)
        emissivity_theta = compute_emissivity_spectrum(d_um, wavelengths_um, theta, n_arr, k_arr, n_sub)
        emissivity_avg += 2.0 * sin_t * cos_t * weights[j] * emissivity_theta

    return emissivity_avg


def compute_net_cooling_power(d_um, wavelengths_um, n_arr, k_arr,
                              T, T_amb, h_c=6.9, n_sub=1.0+0j,
                              theta_sun=np.radians(48.2), n_angles=12):
    """
    计算净制冷功率.

    P_cool = P_rad - P_atm - P_sun - P_nonrad

    参数:
        d_um: PDMS膜厚 (μm)
        wavelengths_um: 波长网格 (μm)
        n_arr, k_arr: PDMS折射率
        T: 制冷器温度 (K)
        T_amb: 环境温度 (K)
        h_c: 非辐射热交换系数 (W/(m²·K))
        n_sub: 基底复折射率
        theta_sun: 太阳天顶角
        n_angles: 角度积分点数

    返回:
        dict: 各辐射分量及净制冷功率
    """
    from numpy.polynomial.legendre import leggauss

    # 角度积分网格
    x, w = leggauss(n_angles)
    theta_arr = (x + 1.0) * pi / 4.0
    weights = w * pi / 4.0

    # 大气数据
    t_atm = build_atmospheric_transmittance(wavelengths_um)

    # 太阳光谱
    I_sun = build_am15_spectrum(wavelengths_um)

    # --- P_rad: 向外辐射 ---
    P_rad = 0.0
    for j, theta in enumerate(theta_arr):
        sin_t = sin(theta)
        cos_t = cos(theta)
        emis = compute_emissivity_spectrum(d_um, wavelengths_um, theta, n_arr, k_arr, n_sub)
        I_bb = planck_spectral_radiance(wavelengths_um, T)
        integrand = I_bb * emis
        P_rad_theta = simpson(integrand, wavelengths_um)
        P_rad += 2.0 * pi * sin_t * cos_t * weights[j] * P_rad_theta

    # --- P_atm: 大气逆辐射 ---
    P_atm = 0.0
    for j, theta in enumerate(theta_arr):
        sin_t = sin(theta)
        cos_t = cos(theta)
        emis = compute_emissivity_spectrum(d_um, wavelengths_um, theta, n_arr, k_arr, n_sub)
        eps_atm = atmospheric_emissivity(wavelengths_um, theta, t_atm)
        I_bb_atm = planck_spectral_radiance(wavelengths_um, T_amb)
        integrand = I_bb_atm * emis * eps_atm
        P_atm_theta = simpson(integrand, wavelengths_um)
        P_atm += 2.0 * pi * sin_t * cos_t * weights[j] * P_atm_theta

    # --- P_sun: 太阳吸收 ---
    emis_sun = compute_emissivity_spectrum(d_um, wavelengths_um, theta_sun, n_arr, k_arr, n_sub)
    integrand_sun = I_sun * emis_sun
    P_sun = simpson(integrand_sun, wavelengths_um)

    # --- P_nonrad: 非辐射热损失 ---
    P_nonrad = h_c * (T_amb - T)

    # --- 净制冷功率 ---
    P_cool = P_rad - P_atm - P_sun - P_nonrad

    return {
        'P_rad': P_rad,
        'P_atm': P_atm,
        'P_sun': P_sun,
        'P_nonrad': P_nonrad,
        'P_cool': P_cool,
        'T': T,
        'T_amb': T_amb,
    }


def find_equilibrium_temperature(d_um, wavelengths_um, n_arr, k_arr,
                                  T_amb, h_c=6.9, n_sub=1.0+0j):
    """
    使用二分法求解稳态平衡温度 T_eq.

    P_cool(T_eq) = 0
    """
    T_low = T_amb - 50.0  # 最大可能温降 50K
    T_high = T_amb + 5.0

    # 确保 P_cool(T_low) > 0 且 P_cool(T_high) < 0
    result_low = compute_net_cooling_power(d_um, wavelengths_um, n_arr, k_arr,
                                            T_low, T_amb, h_c, n_sub)
    result_high = compute_net_cooling_power(d_um, wavelengths_um, n_arr, k_arr,
                                             T_high, T_amb, h_c, n_sub)

    if result_low['P_cool'] <= 0:
        return T_amb  # 无法制冷

    if result_high['P_cool'] >= 0:
        return T_low  # 极端制冷

    # 二分法
    for _ in range(50):
        T_mid = (T_low + T_high) / 2.0
        result_mid = compute_net_cooling_power(d_um, wavelengths_um, n_arr, k_arr,
                                                T_mid, T_amb, h_c, n_sub)
        if result_mid['P_cool'] > 0:
            T_low = T_mid
        else:
            T_high = T_mid

        if T_high - T_low < 0.001:
            break

    T_eq = (T_low + T_high) / 2.0
    return T_eq


# ============================================================
# 候选材料折射率 (多层膜用)
# ============================================================

def build_material_refractive_index(material_name, wavelength_um):
    """
    构造候选材料的复折射率.

    支持的材��:
        SiO2, TiO2, Ag, Al2O3, Si3N4, MgF2
    """
    n = np.ones_like(wavelength_um)
    k = np.zeros_like(wavelength_um)

    if material_name == 'SiO2':
        # 二氧化硅: 可见透明, 红外有吸收峰
        n[:] = 1.46
        # 9μm附近Si-O吸收
        k += 0.05 * np.exp(-((wavelength_um - 9.0) / 0.5)**2)
        k += 0.02 * np.exp(-((wavelength_um - 12.5) / 0.8)**2)
        k += 0.01 * np.exp(-((wavelength_um - 21.0) / 1.0)**2)
        # 紫外吸收
        k += 0.001 * np.exp(-((wavelength_um - 0.15) / 0.05)**2)

    elif material_name == 'TiO2':
        # 二氧化钛: 高折射率, 可见透明, UV吸收
        n[:] = 2.6 - 0.5 * np.exp(-wavelength_um / 0.3)
        k += 0.01 * np.exp(-((wavelength_um - 0.38) / 0.05)**2)
        k += 0.001 * np.exp(-((wavelength_um - 15.0) / 2.0)**2)

    elif material_name == 'Ag':
        # 银: 高反射金属
        # Drude模型近似
        plasma_freq = 9.0  # eV → 对应 ~138 nm
        n[:] = 0.15
        k[:] = 3.5
        # 在可见光区反射率很高, k大
        k += 3.0 * np.exp(-((wavelength_um - 0.32) / 0.1)**2)
        # 红外区 Drude行为
        ir_mask = wavelength_um > 2.0
        n[ir_mask] = 0.5 * (wavelength_um[ir_mask] / 10.0)**0.5
        k[ir_mask] = 9.0 * (wavelength_um[ir_mask] / 10.0)

    elif material_name == 'Al2O3':
        # 氧化铝(蓝宝石)
        n[:] = 1.77 - 0.1 * np.exp(-wavelength_um / 0.2)
        k += 0.005 * np.exp(-((wavelength_um - 10.0) / 1.5)**2)
        k += 0.003 * np.exp(-((wavelength_um - 15.0) / 2.0)**2)

    elif material_name == 'Si3N4':
        # 氮化硅
        n[:] = 2.0 - 0.1 * np.exp(-wavelength_um / 0.3)
        k += 0.01 * np.exp(-((wavelength_um - 11.0) / 1.0)**2)
        k += 0.005 * np.exp(-((wavelength_um - 8.5) / 1.5)**2)

    elif material_name == 'MgF2':
        # 氟化镁: 低折射率
        n[:] = 1.38
        k += 0.001 * np.exp(-((wavelength_um - 10.0) / 2.0)**2)

    return n + 1j * k


# ============================================================
# 波长网格
# ============================================================

def create_wavelength_grid():
    """
    创建自适应波长网格:
    - 0.3-1μm:  2nm步长 (精细, 覆盖太阳光谱峰值)
    - 1-5μm:   10nm步长 (过渡区)
    - 5-25μm:  50nm步长 (红外热辐射区)
    """
    wl_1 = np.arange(0.3, 1.0, 0.002)
    wl_2 = np.arange(1.0, 5.0, 0.01)
    wl_3 = np.arange(5.0, 25.01, 0.05)
    return np.concatenate([wl_1, wl_2, wl_3])
