import pandapower as pp
import pandapower.networks as pn
import pandas as pd

def inspect_case14_physics():
    # 1. 加载标准 Case 14
    net = pn.case14()
    
    print(f"====== IEEE 14 Node Inspection (Total Buses: {len(net.bus)}) ======")
    
    # 2. 建立节点到特殊元件的映射
    special_info = {i: [] for i in net.bus.index}
    
    # --- A. 检查变压器 (Transformers) ---
    # 变压器不仅有阻抗，还有变比 (Tap Ratio) 和 移相角 (Shift)
    # 如果忽略 Tap，电压关系 V_hv = Tap * V_lv 就会算错
    print("\n[1] Transformers (变压器):")
    print(f"{'ID':<4} {'HV_Bus':<8} {'LV_Bus':<8} {'Tap_Pos':<8} {'Shift(deg)':<10} {'Model'}")
    print("-" * 60)
    for idx, row in net.trafo.iterrows():
        hv = row['hv_bus']
        lv = row['lv_bus']
        tap = row['tap_pos']
        shift = row['shift_degree']
        
        info_hv = f"Trafo-HV (to {lv})"
        info_lv = f"Trafo-LV (to {hv})"
        
        # 标记特殊变压器
        if not pd.isna(tap) and tap != 0:
            info_hv += f" [Tap={tap}]"
            info_lv += f" [Tap={tap}]"
        
        special_info[hv].append(info_hv)
        special_info[lv].append(info_lv)
        
        print(f"{idx:<4} {hv:<8} {lv:<8} {str(tap):<8} {shift:<10} {row['std_type']}")

    # --- B. 检查并联元件 (Shunts) ---
    # Shunt 会直接对地注入或吸收无功 Q，导致 Q_inj != Sum(Q_flow)
    print("\n[2] Shunts (并联电容/电抗):")
    print(f"{'ID':<4} {'Bus':<8} {'P(MW)':<10} {'Q(MVar)':<10}")
    print("-" * 60)
    for idx, row in net.shunt.iterrows():
        bus = row['bus']
        q_val = row['q_mvar']
        special_info[bus].append(f"SHUNT (Q={q_val} MVar)")
        print(f"{idx:<4} {bus:<8} {row['p_mw']:<10} {q_val:<10}")

    # --- C. 检查线路充电电容 (Line Charging) ---
    # 你的物理公式忽略了 c_nf_per_km
    print("\n[3] Lines with High Charging (显著对地电容):")
    for idx, row in net.line.iterrows():
        if row['c_nf_per_km'] > 0:
            f, t = row['from_bus'], row['to_bus']
            # 只标记比较大的
            if row['length_km'] * row['c_nf_per_km'] > 100: # 假设阈值
                msg = f"Line-C (to {t})"
                special_info[f].append("High-C Line")
                special_info[t].append("High-C Line")

    # --- D. 汇总输出有问题节点的嫌疑 ---
    print("\n" + "="*60)
    print(">>> 节点成分汇总 (Suspect Analysis) <<<")
    print(f"{'Node':<5} | {'Type':<6} | {'Components Attached'}")
    print("-" * 60)
    
    # 你的 Log 里报错的节点：3, 4, 5, 7, 8, 9
    suspect_nodes = [3, 4, 5, 7, 8, 9]
    
    for i in range(len(net.bus)):
        role = "PQ"
        if i in net.ext_grid.bus.values: role = "Slack"
        elif i in net.gen.bus.values: role = "PV"
        
        comps = ", ".join(special_info[i])
        
        prefix = "  "
        if i in suspect_nodes:
            prefix = "🔴" # 标记为你之前发现误差大的节点
            
        if comps == "":
            comps = "(Pure Line)"
            
        print(f"{prefix} {i:<4} | {role:<6} | {comps}")

if __name__ == '__main__':
    inspect_case14_physics()