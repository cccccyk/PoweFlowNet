import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def sample_and_plot_voltage_extremes():
    print("🔍 正在采样 100 次 IEEE 118（增强版：鼓励轻载过电压）...")
    rng = np.random.default_rng(42)
    
    global_max_voltages = []      # 全局最高电压
    pq_max_voltages = []          # PQ 节点中最高电压
    global_min_voltages = []      # 全局最低电压
    
    for i in range(100):
        net = pn.case118()
        
        # ✅ 调整采样参数以鼓励极端情况
        load_scale = rng.uniform(0.3, 1.0, size=len(net.load))           # 更轻载
        gen_p_scale = rng.uniform(0.8, 1.3, size=len(net.gen))           # 发电可能不足
        net.gen.vm_pu = rng.uniform(0.95, 1.05, size=len(net.gen))       # 略提高上限
        
        net.load.p_mw *= load_scale
        net.load.q_mvar *= load_scale
        net.gen.p_mw *= gen_p_scale
        net.ext_grid.vm_pu = 1.0
        net.ext_grid.a_degree = 0.0
        
        try:
            pp.runpp(net, algorithm='nr', max_iteration=100)
        except:
            continue
        
        vm = net.res_bus.vm_pu
        
        # 全局最高电压
        global_max_v = vm.max()
        global_max_voltages.append(global_max_v)
        
        # PQ 节点最高电压
        pv_buses = set(net.gen.bus.tolist())
        slack_bus = net.ext_grid.bus.iloc[0]
        non_pq = pv_buses | {slack_bus}
        pq_buses = [b for b in net.bus.index if b not in non_pq]
        if pq_buses:
            pq_max_v = vm.loc[pq_buses].max()
            pq_max_voltages.append(pq_max_v)
        
        # 全局最低电压
        global_min_v = vm.min()
        global_min_voltages.append(global_min_v)
    
    print(f"✅ 成功运行 {len(global_max_voltages)} / 100 次")
    
    # 标准 case 参考值
    std_net = pn.case118()
    pp.runpp(std_net)
    std_vm = std_net.res_bus.vm_pu
    std_pv_buses = set(std_net.gen.bus.tolist()) | {std_net.ext_grid.bus.iloc[0]}
    std_pq_buses = [b for b in std_net.bus.index if b not in std_pv_buses]
    std_pq_max = std_vm.loc[std_pq_buses].max()  # ≈1.04292
    std_global_min = std_vm.min()               # ≈0.943

    # === 绘图 ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 图1: PQ 节点最高电压分布（重点看是否 >1.05）
    ax1.hist(pq_max_voltages, bins=15, color='lightcoral', edgecolor='black', alpha=0.7)
    ax1.axvline(1.05, color='red', linestyle='--', label='高压阈值 (1.05)')
    ax1.axvline(std_pq_max, color='green', linestyle=':', label=f'标准PQ最高 ({std_pq_max:.3f})')
    ax1.set_xlabel('PQ 节点最高电压 (p.u.)')
    ax1.set_ylabel('频次')
    ax1.set_title('PQ 节点最高电压分布\n(目标：多于1.05)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 图2: 全局最低电压分布（关注 <0.95）
    ax2.hist(global_min_voltages, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(0.95, color='red', linestyle='--', label='低压阈值 (0.95)')
    ax2.axvline(std_global_min, color='green', linestyle=':', label=f'标准最低 ({std_global_min:.3f})')
    ax2.set_xlabel('全局最低电压 (p.u.)')
    ax2.set_ylabel('频次')
    ax2.set_title('全局最低电压分布\n(关注低压越界)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("voltage_extremes_analysis.png", dpi=150, bbox_inches='tight')
    print("📊 图像已保存为 voltage_extremes_analysis.png")
    plt.show()

if __name__ == "__main__":
    sample_and_plot_voltage_extremes()