import pandapower as pp
import pandapower.networks as nw
import numpy as np
import pandas as pd
from pandapower.pd2ppc import _pd2ppc

# 尝试导入 makeYbus
try:
    from pandapower.pypower.makeYbus import makeYbus
except ImportError:
    from pandapower.pf.makeYbus import makeYbus

def verify_perfect_physics():
    print(">>> 1. 加载 IEEE 14 节点系统并计算潮流...")
    net = nw.case14()
    
    # 运行潮流计算
    pp.runpp(net, algorithm='nr', numba=False)
    
    print(f"   Converged: {net.converged}")

    # -------------------------------------------------------------
    # A. 基于 Ybus 的物理计算 (计算值)
    # -------------------------------------------------------------
    print(">>> 2. 构建 Ybus 并计算 S = V * conj(Y * V)...")
    
    # 1. 转换为 PYPOWER 格式
    ppc, ppci = _pd2ppc(net)
    baseMVA, bus, branch = ppc["baseMVA"], ppc["bus"], ppc["branch"]
    
    # 2. 生成导纳矩阵 Ybus
    # 注意：makeYbus 会自动把 net.shunt (并联电容) 和 线路充电电容 放入 Ybus 对角线
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    
    # 3. 提取收敛后的电压 (Complex Voltage)
    # ppc['bus'] 的第一列是内部节点索引
    bus_idx = bus[:, 0].astype(int)
    
    # 从结果中读取 Vm 和 Va
    vm = net.res_bus.loc[bus_idx, 'vm_pu'].values
    va = net.res_bus.loc[bus_idx, 'va_degree'].values
    
    # 构建复数电压向量
    V_complex = vm * np.exp(1j * np.deg2rad(va))
    
    # 4. 计算注入电流 I = Y * V
    I_inj = Ybus * V_complex
    
    # 5. 计算注入功率 S_calc (标幺值)
    S_calc = V_complex * np.conj(I_inj)

    # -------------------------------------------------------------
    # B. 基于元件汇总的真实注入 (真实值)
    # -------------------------------------------------------------
    print(">>> 3. 手动汇总 Gen 和 Load 计算真实注入 (S_target)...")
    
    # 初始化 P_gen, Q_gen, P_load, Q_load
    n_nodes = len(net.bus)
    P_gen = np.zeros(n_nodes)
    Q_gen = np.zeros(n_nodes)
    P_load = np.zeros(n_nodes)
    Q_load = np.zeros(n_nodes)
    
    # 1. 累加 Load (负载消耗)
    # 注意：Pandapower 中 Load 的 p_mw 是正数表示消耗
    for idx, row in net.load.iterrows():
        bus_id = int(row['bus'])
        # 还要考虑 scaling (如果有的话，这里假设 scaling=1.0)
        scaling = row['scaling'] if 'scaling' in row else 1.0
        P_load[bus_id] += row['p_mw'] * scaling
        Q_load[bus_id] += row['q_mvar'] * scaling

    # 2. 累加 Gen (发电机)
    # 注意：必须用 res_gen，因为 PV 节点的 Q 是算出来的，不是输入的
    for idx, row in net.res_gen.iterrows():
        # 找到对应的 bus
        bus_id = int(net.gen.loc[idx, 'bus'])
        P_gen[bus_id] += row['p_mw']
        Q_gen[bus_id] += row['q_mvar']
        
    # 3. 累加 Ext_Grid (平衡节点)
    # 同样用 res_ext_grid
    for idx, row in net.res_ext_grid.iterrows():
        bus_id = int(net.ext_grid.loc[idx, 'bus'])
        P_gen[bus_id] += row['p_mw']
        Q_gen[bus_id] += row['q_mvar']
        
    # 4. 计算净注入功率 (Target)
    # S_inj = (Sum_Gen - Sum_Load) / BaseMVA
    # 注意：我们不需要减去 Shunt，因为 Shunt 已经被 makeYbus 包含在 Ybus 里了
    # 物理公式 I = YV 计算的是 "流入网络的净电流"，所以对应 (Gen - Load)
    
    P_target = (P_gen - P_load) / baseMVA
    Q_target = (Q_gen - Q_load) / baseMVA
    
    S_target = P_target + 1j * Q_target

    # -------------------------------------------------------------
    # C. 对比验证
    # -------------------------------------------------------------
    print("\n" + "="*85)
    print(f"{'Bus':<5} | {'P_Target':<12} {'P_Calc':<12} {'P_Diff':<10} | {'Q_Target':<12} {'Q_Calc':<12} {'Q_Diff':<10}")
    print("-" * 85)
    
    err_p = np.abs(np.real(S_calc) - P_target)
    err_q = np.abs(np.imag(S_calc) - Q_target)
    
    for i in range(len(bus_idx)):
        # 这里的 i 对应内部排序，通常也是 0~13
        idx = bus_idx[i]
        
        pt = P_target[idx]
        pc = np.real(S_calc[i])
        pd = err_p[i]
        
        qt = Q_target[idx]
        qc = np.imag(S_calc[i])
        qd = err_q[i]
        
        # 标记大误差
        flag = "🔴" if (pd > 1e-5 or qd > 1e-5) else " "
        
        print(f"{idx:<5} | {pt:<12.6f} {pc:<12.6f} {pd:.1e}    | {qt:<12.6f} {qc:<12.6f} {qd:.1e} {flag}")

    print("-" * 85)
    max_err_p = np.max(err_p)
    max_err_q = np.max(err_q)
    print(f"Max P Error: {max_err_p:.4e}")
    print(f"Max Q Error: {max_err_q:.4e}")
    
    if max_err_p < 1e-6 and max_err_q < 1e-6:
        print("\n✅ 验证成功！所有节点的 P 和 Q 都完美匹配。")
        print("结论：")
        print("1. Ybus 包含了所有物理特性 (Line, Trafo, Shunt)。")
        print("2. 注入功率定义为 (Gen - Load)。")
        print("3. 并联元件 (Shunt) 的无功被 Ybus 自动处理了，不需要在 Target 中减去。")
    else:
        print("\n❌ 验证失败，仍有偏差。")

if __name__ == "__main__":
    verify_perfect_physics()