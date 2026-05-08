import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import os

# ==========================================
# 1. 基础设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ==========================================
# 2. 数据处理
# ==========================================
# 请确保文件名或路径正确
file_path = "lam汇总表格简化.xlsx" 
sheet_names = ['TR', 'GS', 'PRC', 'T2S']
data_frames = []

for sheet in sheet_names:
    try:
        df = pd.read_excel(file_path, sheet_name=sheet, header=[0, 1])
        lam_cols = [c[0] for c in df.columns if str(c[0]).startswith('lam1=')]
        lam_groups = sorted(list(set(lam_cols))) 
        
        for idx, row in df.iterrows():
            model_name = row.iloc[0]
            if not isinstance(model_name, str): continue

            for lam in lam_groups:
                def get_val(metrics):
                    for m in metrics:
                        if (lam, m) in df.columns: return row[(lam, m)]
                    return np.nan

                nsfw = get_val(['NSFW'])
                detect = get_val(['Detect_Rate', 'Detection_Rate'])
                bit_acc = get_val(['Bit_Acc'])
                
                # 记录原始数值
                lam_val_str = lam.replace('lam1=', '')
                try:
                    lam_val = float(lam_val_str)
                except:
                    lam_val = 0

                data_frames.append({
                    'Method': sheet,
                    'Model': model_name,
                    'Lam_Raw': lam_val,
                    'NSFW': nsfw,
                    'Detect_Rate': detect,
                    'Bit_Acc': bit_acc
                })
    except Exception as e:
        print(f"Error processing {sheet}: {e}")

df_all = pd.DataFrame(data_frames)
for col in ['NSFW', 'Detect_Rate', 'Bit_Acc']:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

# ==========================================
# 3. 绘图配置
# ==========================================
models = ['SD1.4', 'SD1.5', 'SD2.1']
all_methods = ['TR', 'GS', 'PRC', 'T2S']

# 【修改点1】图例标签补全
param_labels = ['Hyperparameter 1', 'Hyperparameter 2', 'Hyperparameter 3']

# 颜色与纹理
colors_param = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
hatches_param = ['///', '...', 'xxx']

# ==========================================
# 4. 绘图主函数
# ==========================================
def plot_sensitivity_by_model(target_model):
    fig, axes = plt.subplots(1, 3, figsize=(18, 3.5))
    
    metrics_map = {
        'NSFW': 'NCGR',
        'Detect_Rate': 'WDR',
        'Bit_Acc': 'BA'
    }
    
    bar_width = 0.25
    
    df_model = df_all[df_all['Model'] == target_model]
    
    for i, (metric_key, metric_label) in enumerate(metrics_map.items()):
        ax = axes[i]
        
        x_ticks = []
        x_labels = []
        current_x = 0
        
        # 动态决定 Method (BitAcc 不画 TR)
        if metric_key == 'Bit_Acc':
            current_methods = [m for m in all_methods if m != 'TR']
        else:
            current_methods = all_methods
            
        for method in current_methods:
            subset_m = df_model[df_model['Method'] == method]
            unique_lams = sorted(subset_m['Lam_Raw'].unique())
            
            group_center = current_x + (len(unique_lams) * bar_width) / 2 - (bar_width / 2)
            x_ticks.append(group_center)
            x_labels.append(method)
            
            for p_idx, lam_val in enumerate(unique_lams):
                if p_idx >= 3: break 
                
                row = subset_m[subset_m['Lam_Raw'] == lam_val]
                val = 0
                if not row.empty:
                    val = row[metric_key].values[0]
                    if pd.isna(val): val = 0
                
                ax.bar(current_x, val, bar_width, 
                       color=colors_param[p_idx], hatch=hatches_param[p_idx], 
                       edgecolor='black', alpha=0.9)
                
                current_x += bar_width
            
            current_x += 0.3
            
        # 样式调整
        ax.set_ylim(0, 1.2)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        ax.set_xticks(x_ticks)
        # 【修改点2】取消加粗 (fontweight 默认为 normal)
        ax.set_xticklabels(x_labels, fontsize=13) 
        
        letters = ['a', 'b', 'c']
        ax.set_title(f'({letters[i]}) {metric_label}', y=-0.28, fontweight='bold', fontsize=14)
        
        if i == 0:
            ax.set_ylabel('Score / Rate')
        ax.set_xlabel('')

    # ==========================================
    # 图例
    # ==========================================
    legend_handles = []
    for idx, label in enumerate(param_labels):
        legend_handles.append(mpatches.Patch(
            facecolor=colors_param[idx], hatch=hatches_param[idx], 
            edgecolor='black', alpha=0.9, label=label
        ))
    
    plt.subplots_adjust(bottom=0.32, wspace=0.15)
    
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, 
               bbox_to_anchor=(0.5, 0.02), fontsize=13, frameon=True, edgecolor='black')
    
    filename = f'Sensitivity_{target_model}.pdf'
    plt.savefig(filename, dpi=300, format='pdf', bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()

# 执行生成
for m in models:
    plot_sensitivity_by_model(m)