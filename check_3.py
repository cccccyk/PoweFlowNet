# verify_ybus.py
import pandapower as pp
import pandapower.networks as nw
import numpy as np
from pandapower.pd2ppc import _pd2ppc

def verify_ybus(case_name='case14'):
    print(f"🧪 Verifying Ybus construction for {case_name}...\n")

    # 1. 加载网络
    if case_name == 'case14':
        net = nw.case14()
    elif case_name == 'case118':
        net = nw.case118()
    else:
        raise ValueError("Only case14 or case118 supported for now.")

    # 【可选】彻底移除所有并联元件（取消注释以测试“纯串联”模型）
    net.shunt.drop(net.shunt.index, inplace=True)
    net.line['c_nf_per_km'] = 0.0  # 关键：清零线路电容


    # 运行潮流（确保 ppc 可生成）
    pp.runpp(net, algorithm='nr', numba=False)
    print(f"✅ Power flow converged: {net.converged}")

    # 2. 获取 ppc 数据
    ppc, _ = _pd2ppc(net)
    bus = ppc['bus']          # shape (n_bus, 13+)
    branch = ppc['branch']    # shape (n_branch, 13)

    # 3. 构建 Ybus
    try:
        from pandapower.pypower.makeYbus import makeYbus
    except ImportError:
        from pandapower.pf.makeYbus import makeYbus

    Ybus, _, _ = makeYbus(ppc['baseMVA'], bus, branch)

    # 4. 建立总线编号到索引的映射
    bus_id_to_idx = {int(bus[i, 0]): i for i in range(len(bus))}
    n_bus = len(bus)

    # 5. 手动计算每个 Yii
    Y_ii_manual = np.zeros(n_bus, dtype=complex)

    print("🔍 Accumulating contributions from branches...")
    for k in range(branch.shape[0]):
        f_bus = int(branch[k, 0])
        t_bus = int(branch[k, 1])
        r = branch[k, 2]
        x = branch[k, 3]
        b_c = branch[k, 4]  # 总充电电纳 (p.u.)

        i = bus_id_to_idx[f_bus]
        j = bus_id_to_idx[t_bus]

        # 串联导纳
        if r == 0 and x == 0:
            y_series = 0.0 + 0.0j
        else:
            y_series = 1.0 / (r + 1j * x)

        # 每端加：y_series + j*(b_c / 2)
        Y_ii_manual[i] += y_series + 1j * (b_c / 2)
        Y_ii_manual[j] += y_series + 1j * (b_c / 2)

    # 6. 加上显式并联元件（来自 bus 表的第 4、5 列）
    print("🔌 Adding explicit shunt elements from bus table...")
    for i in range(n_bus):
        g_shunt = bus[i, 4]  # 并联电导 (p.u.)
        b_shunt = bus[i, 5]  # 并联电纳 (p.u.)
        Y_ii_manual[i] += g_shunt + 1j * b_shunt

    # 7. 对比 Ybus 对角线 vs 手动计算
    print("\n" + "="*80)
    print(f"{'Bus':<5} | {'Ybus_ii (real)':<15} {'Ybus_ii (imag)':<15} | {'Manual_ii (real)':<15} {'Manual_ii (imag)':<15} | {'Max Diff'}")
    print("-"*80)

    max_diff = 0.0
    for i in range(n_bus):
        ybus_val = Ybus[i, i]
        manual_val = Y_ii_manual[i]
        diff = abs(ybus_val - manual_val)
        max_diff = max(max_diff, diff)

        print(f"{int(bus[i,0]):<5} | {ybus_val.real:<15.6f} {ybus_val.imag:<15.6f} | "
              f"{manual_val.real:<15.6f} {manual_val.imag:<15.6f} | {diff:.2e}")

    print("-"*80)
    print(f"🎯 Max absolute difference: {max_diff:.2e}")

    if max_diff < 1e-12:
        print("✅ SUCCESS: Ybus diagonal matches manual calculation!")
    else:
        print("❌ FAILURE: Mismatch detected. Check branch or shunt data.")

    # 8. 额外信息：是否有非零 shunt？
    total_line_bc = np.sum(branch[:, 4])
    total_bus_shunt = np.sum(bus[:, 4:6])
    print(f"\nℹ️  Total line charging susceptance (b_c): {total_line_bc:.6f} p.u.")
    print(f"ℹ️  Total explicit bus shunt (g + jb): {total_bus_shunt.real:.6f} + j{total_bus_shunt.imag:.6f} p.u.")


if __name__ == "__main__":
    # 可切换为 'case118'
    verify_ybus('case14')