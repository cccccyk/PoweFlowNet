import pandapower as pp
import pandapower.networks as pn
import pandas as pd

def verify_cleaning_logic():
    print(">>> 正在加载原始 IEEE 14 节点系统...")
    net = pn.case14()
    
    print(f"\n[原始状态]")
    print(f"Shunt 元件数量: {len(net.shunt)}")
    print(f"线路最大电容 (c_nf): {net.line['c_nf_per_km'].max()}")
    print(f"变压器最大变比 (tap): {net.trafo['tap_pos'].abs().max() if 'tap_pos' in net.trafo else 'None'}")
    
    print("\n>>> 执行净化操作 (模拟 generate_data)...")
    
    # 1. 移除 Shunt
    if not net.shunt.empty:
        net.shunt.drop(net.shunt.index, inplace=True)
        
    # 2. 移除线路电容
    net.line['c_nf_per_km'] = 0.0
    
    # 3. 归零变压器
    if 'tap_pos' in net.trafo.columns:
        net.trafo['tap_pos'] = 0.0
    if 'shift_degree' in net.trafo.columns:
        net.trafo['shift_degree'] = 0.0
        
    print("\n[净化后状态]")
    print(f"Shunt 元件数量: {len(net.shunt)}  <-- 应该是 0")
    print(f"线路最大电容: {net.line['c_nf_per_km'].max()} <-- 应该是 0.0")
    print(f"变压器最大变比: {net.trafo['tap_pos'].abs().max()} <-- 应该是 0.0")
    
    if len(net.shunt) == 0 and net.line['c_nf_per_km'].sum() == 0:
        print("\n✅ 验证通过！代码逻辑可以成功清除复杂物理特性。")
    else:
        print("\n❌ 验证失败！仍有残留。")

if __name__ == "__main__":
    verify_cleaning_logic()