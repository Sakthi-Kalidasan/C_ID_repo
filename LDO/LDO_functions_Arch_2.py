## Python script with functions to estimate small signal feasibility of an LDO design
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def estimate_ldo_power_and_pass_params(
    PSRR_target_dB, f_bw, PM_target, I_load, I_res_div, V_out, V_ctrl, C_out, 
    csv_file_pmos_stage2, csv_file_pass_pmos, kgm_target=25
):
    """
    Combines PMOS pass device estimation and amplifier stage power estimation.

    Returns:
        result: dict containing PMOS pass parameters and amplifier stage estimation results.
    """
    # Estimate PMOS pass device parameters
    C_in_pass, Av_pass, k_res_div, details_pass = estimate_pass_pmos_parameters(
        csv_file_pass_pmos, I_load, I_res_div, V_out, V_ctrl, f_bw, C_out
    )
    # Use C_in_pass as C_load for amplifier stage estimation
    VDC_target = 1.25  # Target DC voltage at the output of the amplifier determined by the LDO pass transistor
    kgm2_target = 4
    kgm1_target = 25
    I_total, I_req_stage1, I_req_stage2, GBW_stage1, GBW_stage2 = estimate_GBW_stages(PSRR_target_dB, f_bw, PM_target, k_res_div, Av_pass, csv_file_pmos_stage2, VDC_target, C_in_pass, kgmp_2_target=kgm2_target, kgmp_stage1=kgm1_target)

    result = {
        "C_in_pass": C_in_pass,
        "Av_pass": Av_pass,
        "k_res_div": k_res_div,
        "details_pass": details_pass,
        "GBW_stage1": GBW_stage1,
        "GBW_stage2": GBW_stage2,
        "I_total": I_total,
        "I_req1": I_req_stage1,
        "I_req2": I_req_stage2,
    }
    return result

def estimate_stage2_cs_amp_with_res_load_params(csv_file, VDC_target, C_Load,f_p, kgmp_target):
    """
    Estimate the required current (I_req), voltage gain (Av), and input capacitance (Cin)
    for a PMOS amplifier stage given a target DC voltage, load capacitance, and pole frequency.

    Args:
        csv_file (str): Path to the CSV file with device parameters.
        VDC_target (float): Target DC voltage at the output of the amplifier (V).
        C_Load (float): Load capacitance (F).
        f_p (float): Target pole frequency (Hz).

    Returns:
        tuple: (I_req, Av, Cin)
            I_req (float): Required current (A).
            Av (float): Voltage gain at the selected point.
            Cin (float): Input capacitance (F).
    """

    param_master = pd.read_csv(csv_file)
    # Extract relevant columns
    kgm_p = np.abs(param_master['kgmp Y'])
    kcgs_p = np.abs(param_master['kcsg_p Y'])
    kcgd_p = np.abs(param_master['kcdg_p Y'])
    kgds_p = np.abs(param_master['kgds_p Y'])
    ids = np.abs(param_master['kgmp X'])

    # Calculate effective resistance for the target pole
    R_eff = 1 / (2 * np.pi * f_p * C_Load)
    # Required current for the target DC voltage
    I_req = VDC_target / R_eff

    # Find the index where kgm_p is closest to the target kgmp
    # Add kgmp_target as an input argument to the function
    # (You must also add kgmp_target to the function signature above)
    idx = (kgm_p - kgmp_target).abs().idxmin()
    gm_p = kgm_p.iloc[idx] * I_req
    gds_p = kgds_p.iloc[idx] * I_req
    Av = gm_p / gds_p if gds_p != 0 else np.nan

    cgs_p = kcgs_p.iloc[idx] * I_req
    cgd_p = kcgd_p.iloc[idx] * I_req
    Cin = cgs_p + cgd_p * (1 + Av)

    return I_req, Av, Cin


def estimate_GBW_stages(PSRR_target_dB, f_bw, PM_target, k_res_div=3/4, Av_pass=25, csv_file=None, VDC_target=1.25, C_Load=11e-12, kgmp_2_target=15, kgmp_stage1=25):
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
    f_p = f_p3        # Use calculated second pole frequency

    # Only call if csv_file is provided
    if csv_file is not None:
        I_req_stage2, Av_2, Cin_stage2 = estimate_stage2_cs_amp_with_res_load_params(csv_file, VDC_target, C_Load, f_p, kgmp_2_target)
    else:
        Av_2 = 1  # Fallback if no CSV file provided


    GBW_stage2 = Av_2 * f_p3 #Assuming third pole at second stage

    Av_1 = (1 / PSRR_target) * (1 / Av_2) * (1/k_res_div) * (1/Av_pass)
    GBW_stage1 = Av_1 * f_p2 #Assuming second pole at first stage

    I_req_stage1 = GBW_stage1*Cin_stage2 / kgmp_stage1
    I_total = I_req_stage1 + I_req_stage2

    return I_total,I_req_stage1,I_req_stage2, GBW_stage1, GBW_stage2

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


## Testing the script 

if __name__ == "__main__": 
    # Example parameters
    PSRR_target_dB = -40  # Target PSRR in dB
    f_bw = 100e3  # Desired bandwidth in Hz
    PM_target = 60  # Target phase margin in degrees
    I_load = 10e-3  # Load current in A
    I_res_div = 5e-6  # Resistor divider current in A
    V_out = 1.2  # Output voltage in V
    V_ctrl = 1.25  # Control voltage in V
    C_out = 10e-12  # Output capacitance in F
    csv_file_ldo_pass = 'LDO/ldo_pmos_kgm_char_Msweep_10mA.csv'  # Path to NMOS parameters CSV file
    csv_file_pmos_stage2 = 'LDO/cs_amp_pmos_master.csv'  # Path to PMOS parameters CSV file

    result = estimate_ldo_power_and_pass_params(
        PSRR_target_dB, f_bw, PM_target, I_load, I_res_div, V_out, V_ctrl, C_out,
       csv_file_pmos_stage2, csv_file_ldo_pass,  kgm_target=25
    )

    print("\nEstimated LDO Parameters:")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4e}")
        else:
            print(f"{key}: {value}")
