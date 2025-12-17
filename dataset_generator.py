import pandapower as pp
import pandapower.networks as nw
import numpy as np
from pandapower.pd2ppc import _pd2ppc

# --- 兼容导入 ---
try:
    from pandapower.pypower.makeYbus import makeYbus
except ImportError:
    from pandapower.pf.makeYbus import makeYbus

def verify_correctly():
    # 1. 加载网络并计算
    net = nw.case118()

    # 1. 移除并联元件 (Shunts)
        # 彻底删除 shunt 表中的所有行
    if not net.shunt.empty:
        net.shunt.drop(net.shunt.index, inplace=True) 

    pp.runpp(net, algorithm='nr', numba=False)
    print(f"Convergence: {net.converged}")

    # 2. 获取 Ybus
    # 注意：ppc 里的 bus 顺序可能跟 net.bus 的 index 顺序不一样 (虽然 case14 通常一样)
    # 我们必须确保对齐。
    ppc, ppci = _pd2ppc(net)
    baseMVA, bus, branch = ppc["baseMVA"], ppc["bus"], ppc["branch"]
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    print(Ybus)
    
    # 3. 【关键修正】从 net.res_bus 提取收敛后的电压
    # 我们需要按照 ppc['bus'] 的顺序来提取电压
    # ppc['bus'] 的第 0 列 (BUS_I) 是总线索引
    bus_indices = bus[:, 0].astype(int)
    
    # 从 res_bus 中提取对应的 Vm 和 Va
    # 注意：pandapower 的索引通常对应 bus_indices
    vm_converged = net.res_bus.loc[bus_indices, 'vm_pu'].values
    va_converged_deg = net.res_bus.loc[bus_indices, 'va_degree'].values + np.random.uniform(0, 0.02, size=118)
    
    # 构建复数电压向量
    V_calc = vm_converged * np.exp(1j * np.deg2rad(va_converged_deg))
    
    # 4. 计算注入功率 (Physics)
    # S_inj = V * conj(Y * V)
    # 注意：这里计算的是标幺值 (p.u.)
    I_inj = Ybus * V_calc
    S_calc_pu = V_calc * np.conj(I_inj)
    
    # 5. 【关键修正】从 net.res_bus 提取真实功率
    # Pandapower res_bus 定义:
    # p_mw > 0: 节点从电网吸收功率 (Load)
    # p_mw < 0: 节点向电网注入功率 (Gen)
    # 
    # 物理公式 S_inj 是 "注入电网的功率"。
    # 所以: S_physics = -1 * (res_bus_p + j*res_bus_q)
    
    p_res_mw = net.res_bus.loc[bus_indices, 'p_mw'].values
    q_res_mvar = net.res_bus.loc[bus_indices, 'q_mvar'].values
    
    # 转换为 p.u. 且取反 (转为注入方向)
    P_data_pu = -p_res_mw / net.sn_mva
    Q_data_pu = -q_res_mvar / net.sn_mva
    
    # 6. 验证对比
    err_p = np.abs(np.real(S_calc_pu) - P_data_pu)
    err_q = np.abs(np.imag(S_calc_pu) - Q_data_pu)
    
    print("\n" + "="*60)
    print(f"{'Bus':<5} | {'P_Data (res_bus)':<18} {'P_Calc (Y*V_conv)':<18} | {'Diff':<10}")
    print("-" * 60)
    
    for i in range(len(bus_indices)):
        idx = bus_indices[i]
        p_d = P_data_pu[i]
        p_c = np.real(S_calc_pu[i])
        diff = err_p[i]
        diff_q = err_q[i]
        rate_p = err_p[i] / p_d * 100
        rate_q = err_q[i] / Q_data_pu[i] * 100
        print(f"{idx:<5} | {p_d:<18.6f} {p_c:<18.6f} | {diff:.8f} | {diff_q:.8f} | {rate_p:.8f} | {rate_q:.8f}")

    print("-" * 60)
    print(f"Max P Error: {np.max(err_p):.2e}")
    print(f"Max Q Error: {np.max(err_q):.2e}")

    if np.max(err_p) < 1e-8:
        print("\n✅ PERFECT MATCH! (Finally)")
    else:
        print("\n❌ STILL WRONG.")

if __name__ == "__main__":
    verify_correctly()