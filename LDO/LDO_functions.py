## Python script with functions to estimate small signal feasibility of an LDO design
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def estimate_gbw_and_amplifier_stages(PSRR_target_dB, f_bw, PM_target, C_load, csv_file, kgm_target=25):
    """
    Combines GBW estimation and amplifier stage estimation.

    Returns:
        result: dict containing GBW_stage1, GBW_stage2, and amplifier stage estimation results.
    """
    GBW_stage1, GBW_stage2 = estimate_GBW_stages(PSRR_target_dB, f_bw, PM_target, C_load)
    amp_result = estimate_amplifier_stages(csv_file, GBW_stage1, GBW_stage2, C_load, kgm_target=kgm_target)
    result = {
        "GBW_stage1": GBW_stage1,
        "GBW_stage2": GBW_stage2,
        "amplifier_stages": amp_result
    }
    return result

def analyze_second_stage(C_load, GBW_stage2, kgm_max=25):
    gm_req = 2 * np.pi * C_load * GBW_stage2
    I_req = gm_req / kgm_max
    # print(f"Required gm of the 2nd stage of the amplifier for GBW_stage2 {GBW_stage2*1e-9:.2f} GHz and C_load {C_load*1e12:.1f} pF: {gm_req*1e3:.2f} mS")
    # print(f"Required current of the 2nd stage of the amplifier for kgm_max {kgm_max} : {I_req*1e3:.2f} mA while having I_budget {I_budget*1e3:.1f} mA and PSRR_target {PSRR_target_dB} dB")
    return I_req

def estimate_C_in_stg2(csv_file, I_req, kgm_target):
    """
    Estimate the input capacitance (C_in) for a source follower stage
    by finding the closest kgm in the data and using the corresponding kcgs and kcgd.

    Args:
        csv_file (str): Path to the CSV file with device parameters.
        I_req (float): Required current (A).
        kgm_target (float): Target kgm value.

    Returns:
        float: Estimated input capacitance (F).
        dict: Dictionary with details at the selected point.
    """
    df = pd.read_csv(csv_file)
    kgm_n = np.abs(df['kgmn Y'])
    kcsg_n = np.abs(df['kcgs_n Y'])
    kcgd_n = np.abs(df['kcgd_n Y'])

    idx_kgm = (kgm_n - kgm_target).abs().idxmin()
    ksg_at_kgm = kcsg_n.iloc[idx_kgm]
    kcgd_at_kgm = kcgd_n.iloc[idx_kgm]

    cgs = ksg_at_kgm * I_req
    cgd = kcgd_at_kgm * I_req
    # For a source follower, Av_2 = 1 (as in your notebook)
    C_in = cgs + cgd * (1 + 1)

    details = {
        'idx_kgm': idx_kgm,
        'kgm': kgm_n.iloc[idx_kgm],
        'kcsg_n': ksg_at_kgm,
        'kcgd_n': kcgd_at_kgm,
        'cgs': cgs,
        'cgd': cgd,
        'C_in': C_in
    }
    return C_in, details

def extract_I_req_stage1(C_out, GBW_stage1, kgm_max=25):
    """
    Estimate the required current for the first stage of the amplifier
    given the output capacitance, GBW, and maximum kgm.

    Args:
        C_in (float): Input capacitance of the first stage (F).
        GBW_stage1 (float): Gain-bandwidth product for the first stage (Hz).
        kgm_max (float): Maximum allowed gm/I (default: 25 S/A).

    Returns:
        float: Required current for the first stage (A).
    """
    gm_req = 2 * np.pi * C_out * GBW_stage1
    I_req = gm_req / kgm_max
    return I_req


def estimate_amplifier_stages(csv_file, GBW_stage1, GBW_stage2, C_load, kgm_target=25):
    """
    Estimate the required currents for both amplifier stages and total current.

    Args:
        csv_file (str): Path to the CSV file with device parameters.
        GBW_stage1 (float): Gain-bandwidth product for the first stage (Hz).
        GBW_stage2 (float): Gain-bandwidth product for the second stage (Hz).
        I_budget (float): Current budget for the second stage (A).
        kgm_target (float): Target kgm value for both stages (default: 25).
        C_load (float): Load capacitance for stage 2 (F).

    Returns:
        dict: {
            'I_req_stage2': Required current for stage 2 (A),
            'C_in_stage2': Estimated input capacitance for stage 2 (F),
            'I_req_stage1': Required current for stage 1 (A),
            'I_req_total': Total required current (A),
            'details_stage2': Details at selected point for stage 2
        }
    """
    # 1. Estimate I_req2 based on GBW_stage2 and I_budget
    I_req_stage2 = analyze_second_stage(C_load, GBW_stage2, kgm_target)
    # 2. Estimate Cin for the second stage
    C_in_stage2, details_stage2 = estimate_C_in_stg2(csv_file, I_req_stage2, kgm_target)
    # 3. Estimate I_req1 based on Cin and GBW_stage1
    I_req_stage1 = extract_I_req_stage1(C_in_stage2, GBW_stage1, kgm_target)
    # 4. Total required current
    I_req_total = I_req_stage1 + I_req_stage2

    print(f"Stage 2 required current (I_req_stage2): {I_req_stage2*1e3:.4f} mA")
    print(f"Stage 2 input capacitance (C_in_stage2): {C_in_stage2*1e12:.2f} pF")
    print(f"Stage 1 required current (I_req_stage1): {I_req_stage1*1e3:.4f} mA")
    print(f"Total required current (I_req_total): {I_req_total*1e3:.4f} mA")
    return {
        'I_req_stage2': I_req_stage2,
        'C_in_stage2': C_in_stage2,
        'I_req_stage1': I_req_stage1,
        'I_req_total': I_req_total,
        'details_stage2': details_stage2
    }

def estimate_GBW_stages(PSRR_target_dB, f_bw, PM_target, Av_2=1, k_res_div=3/4, Av_pass=25):
    """
    Estimate the required GBW for stage 1 and stage 2 of the amplifier.

    Args:
        PSRR_target_dB (float or int): PSRR target in dB (negative value).
        f_bw (float): Desired closed-loop bandwidth (Hz).
        PM_target (float or int): Phase margin target (degrees).
        C_load (float): Load capacitance for stage 2 (F).
        Av_2 (float): Gain of stage 2 (default: 1).
        k_res_div (float): Resistive divider ratio (default: 3/4).

    Returns:
        tuple: (GBW_stage1, GBW_stage2) in Hz
    """
    # Convert PSRR from dB to linear
    PSRR_target = 10**(PSRR_target_dB / 20)
    # Unity-gain frequency for the amplifier
    f_u = (1 / PSRR_target) * f_bw
    # Second pole location for phase margin
    f_p2 = f_u * np.tan(np.radians(PM_target))
    # Third pole location for stability
    f_p3 = f_p2 * 2

    # Stage 2 gain (Av_2) is determined by system requirements; here, assume unity due to architecture choice
    Av_2 = 1  # If you want to use a specific Av_1, pass as argument

    Av_1 = (1 / PSRR_target) * (1 / Av_2) * (1/k_res_div) * (1/Av_pass)

    GBW_stage1 = Av_1 * f_p2 #Assuming second pole at first stage
    GBW_stage2 = Av_2 * f_p3 #Assuming third pole at second stage

    return GBW_stage1, GBW_stage2

def estimate_pass_pmos_parameters(csv_file, I_load, I_res_div, Vout_LDO, Vctrl, f_bw, C_out):
    """
    Estimate the effective input capacitance (including Miller effect) of the PMOS pass transistor.

    Args:
        csv_file (str): Path to the CSV file with device parameters.
        I_load (float): Load current (A).
        I_res_div (float): Current through the resistor divider (A).
        Vout_LDO (float): Output voltage of the LDO (V).
        Vctrl (float): Control voltage for error amplifer (V).
        f_bw (float): Bandwidth frequency (Hz).
        C_out (float): Output capacitance (F).

    Returns:
        float: Effective input capacitance (F).
        dict: Details at the selected operating point.
    """
    param_master = pd.read_csv(csv_file)

    kgm_p = np.abs(param_master['kgmp Y'])
    kcsg_p = np.abs(param_master['kcgs Y'])
    kcgd_p = np.abs(param_master['kcgd Y'])
    kgds_p = np.abs(param_master['kgds Y'])
    ids = np.abs(param_master['ids Y'])
    M_scale = np.abs(param_master['kgmp X'])
    vd = np.abs(param_master['vd Y'])


    R_res_div = Vout_LDO / I_res_div
    print(f"Resistor Divider Resistance (R_res_div) for 1% current overhead at I_max: {R_res_div:.2f} Ohms")

    I_total = I_load + I_res_div
    print(f"Total current through the PMOS pass transistor (I_total): {I_total*1e3:.2f} mA")
    
    # Find index where both ids is closest to I_total and vd is closest to 1.2V
    idx_both = ((ids - I_total).abs() + (vd - 1.2).abs()).idxmin()
    M_scale_both = M_scale.loc[idx_both]

    # Interpolate caps at M_scale_both
    kcgd_interp = interp1d(M_scale, kcgd_p, kind='linear', fill_value='extrapolate')
    cgd_p_both = float(kcgd_interp(M_scale_both)) * I_total
    print(f"Gate-Drain Capacitance (Cgd) at I_total {I_total*1e3:.2f} mA: {cgd_p_both*1e12:.3f} pF")
    kcgs_interp = interp1d(M_scale, kcsg_p, kind='linear', fill_value='extrapolate')
    cgs_p_both = float(kcgs_interp(M_scale_both)) * I_total
    print(f"Gate-Source Capacitance (Cgs) at I_total {I_total*1e3:.2f} mA: {cgs_p_both*1e12:.3f} pF")
    # Interpolate gm and gds at M_scale_both
    kgm_interp = interp1d(M_scale, kgm_p, kind='linear', fill_value='extrapolate')
    gm_p_both = float(kgm_interp(M_scale_both)) * I_total
    kgds_interp = interp1d(M_scale, kgds_p, kind='linear', fill_value='extrapolate')
    gds_p_both = float(kgds_interp(M_scale_both)) * I_total

    Av_pass_both = gm_p_both / gds_p_both
    R_L_eff = 1 / (1/R_res_div + gds_p_both)
    print(f"Effective Load Resistance (R_L_eff) considering Rds: {R_L_eff:.2f} Ohms")

    C_load_eff = C_out + cgd_p_both * (1 + (1 / Av_pass_both))  # Effective load capacitance including Miller effect
    print(f"Effective Load Capacitance at I_min: {C_load_eff*1e12:.3f} pF")

    # Estimate R_L required to meet the bandwidth target
    R_L_bw = 1 / (2 * np.pi * f_bw * C_load_eff)
    print(f"Load Resistance (R_L) required for bandwidth target {f_bw:.1f} Hz: {R_L_bw:.2f} Ohms")

    # Min current overhead for resistor divider
    R_res_div_max = max(R_res_div, R_L_bw)
    print(f"Using Resistor Divider Resistance (R_res_div): {R_res_div_max:.2f} Ohms")

    # Estimating the gain with the right R_eff
    R_L_eff = 1 / (1/R_res_div_max + gds_p_both)
    Av_pass = gm_p_both * R_L_eff
    print(f"Av_pass at I_total {I_total*1e3:.2f} mA: {Av_pass:.2f} V/V")


    # Effective input capacitance including Miller effect
    C_in_pass = cgs_p_both + cgd_p_both * (1 + Av_pass)

    k_res_div = Vctrl/Vout_LDO

    details = {
        'idx_both': idx_both,
        'M_scale_both': M_scale_both,
        'gm_p_both': gm_p_both,
        'gds_p_both': gds_p_both,
        'Av_pass': Av_pass,
        'cgs_p_both': cgs_p_both,
        'cgd_p_both': cgd_p_both,
        'C_in_pass': C_in_pass,
        'k_res_div': k_res_div
    }
    return C_in_pass, Av_pass, k_res_div, details
