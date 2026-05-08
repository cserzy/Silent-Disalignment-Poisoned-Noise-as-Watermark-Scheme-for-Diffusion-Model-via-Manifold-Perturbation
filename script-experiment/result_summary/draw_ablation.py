import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import os

# ==========================================
# 1. 基础设置 (学术风格)
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
# 2. 数据读取
# ==========================================
# 使用你的绝对路径
file_path = "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/result_summary/第三批实验汇总-消融实验-2026年1月24日.xlsx"

try:
    # 读取第一个 Sheet
    df = pd.read_excel(file_path, sheet_name=0, header=[0, 1])
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()

# 清洗表头
new_cols = []
for i, col in enumerate(df.columns):
    c1_name = str(col[1]).strip()
    if i == 0:
        new_cols.append(('Meta', 'Model'))
    elif i == 1:
        new_cols.append(('Meta', 'Method'))
    elif 2 <= i <= 4:
        new_cols.append(('w_att', c1_name))
    elif 5 <= i <= 7:
        new_cols.append(('delssc', c1_name))
    elif 8 <= i <= 10:
        new_cols.append(('delrepair', c1_name))
    else:
        new_cols.append(col)

df.columns = pd.MultiIndex.from_tuples(new_cols)
df[('Meta', 'Model')] = df[('Meta', 'Model')].ffill()

# 转换数值 (包含 Detection_Rate 适配)
raw_cats = ['w_att', 'delssc', 'delrepair']
metrics = ['NSFW', 'Detection_Rate', 'Bit_Acc'] 

for cat in raw_cats:
    for metric in metrics:
        df[(cat, metric)] = pd.to_numeric(df[(cat, metric)], errors='coerce')

# ==========================================
# 3. 绘图配置
# ==========================================
models = ['SD1.4', 'SD1.5', 'SD2.1']

label_map = {
    'w_att': 'Ours',
    'delssc': 'w/o SSP',
    'delrepair': 'w/o ARR'
}

colors = {
    'w_att': '#1f77b4',      # Blue
    'delssc': '#d62728',     # Red
    'delrepair': '#2ca02c'   # Green
}
hatches = {
    'w_att': '///',
    'delssc': '...',
    'delrepair': 'xxx'
}

# ==========================================
# 4. 通用横向绘图函数 (修复图例冲突版)
# ==========================================
def plot_metric_row(metric_name, method_list, output_filename):
    n_plots = len(method_list)
    
    # 保持扁平比例：高度 3.2
    fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 3.2))
    
    if n_plots == 1:
        axes = [axes]
    
    x = np.arange(len(models))
    width = 0.25

    for i, method in enumerate(method_list):
        ax = axes[i]
        subset = df[df[('Meta', 'Method')] == method].copy()
        subset['Model_Sort'] = pd.Categorical(subset[('Meta', 'Model')], categories=models, ordered=True)
        subset = subset.sort_values('Model_Sort')
        
        for j, cat in enumerate(raw_cats):
            vals = np.nan_to_num(subset[(cat, metric_name)].values, nan=0.0)
            offset = (j - 1) * width
            ax.bar(x + offset, vals, width, 
                   color=colors[cat], edgecolor='black', hatch=hatches[cat], alpha=0.9, label=label_map[cat])

        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.2) 
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # 子标题
        letters = ['a', 'b', 'c', 'd']
        ax.set_xlabel(f'({letters[i]}) {method}', fontsize=16, fontweight='bold', labelpad=8)
        ax.set_title('')
        
        # Y轴标签
        if i == 0:
            if metric_name == 'NSFW':
                ax.set_ylabel('NSFW Score')
            elif metric_name == 'Detection_Rate':
                ax.set_ylabel('Detection Rate')
            else:
                ax.set_ylabel('Bit Accuracy')

    # 图例 Handles
    legend_handles = []
    for cat in raw_cats:
        legend_handles.append(mpatches.Patch(
            facecolor=colors[cat], edgecolor='black', hatch=hatches[cat], alpha=0.9, 
            label=label_map[cat]
        ))

    # 【核心修改点】调整底部边距和锚点
    # bottom=0.38: 给底部预留更多空间 (针对扁图优化)
    # bbox_to_anchor=(0.5, 0.01): 让图例更贴近最底边缘
    plt.subplots_adjust(bottom=0.38, wspace=0.2)
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, 
               frameon=True, edgecolor='black', bbox_to_anchor=(0.5, 0.01), fontsize=13)

    # 保存 (使用 tight 裁剪掉多余空白)
    plt.savefig(output_filename, dpi=300, format='pdf', bbox_inches='tight')
    print(f"Saved {output_filename}")
    plt.close()

# ==========================================
# 5. 执行
# ==========================================

plot_metric_row('NSFW', ['TR', 'GS', 'PRC', 'T2S'], 'Metric_NSFW_1x4.pdf')
plot_metric_row('Detection_Rate', ['TR', 'GS', 'PRC', 'T2S'], 'Metric_Detect_1x4.pdf')
plot_metric_row('Bit_Acc', ['GS', 'PRC', 'T2S'], 'Metric_BitAcc_1x3.pdf')