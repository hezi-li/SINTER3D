import re
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


class SmartLayerAdapter:
    """
    智能层名适配器 - 自动检测各种层命名格式并进行标准化
    """
    
    def __init__(self):
        self.layer_patterns = [
            r'Layer_(\d+)',      # Layer_1, Layer_2, ...
            r'Layer(\d+)',       # Layer1, Layer2, ...
            r'L(\d+)',           # L1, L2, ...
            r'layer_(\d+)',      # layer_1, layer_2, ...
            r'layer(\d+)',       # layer1, layer2, ...
            r'(\d+)',            # 1, 2, 3, ...
        ]
    
    def detect_layer_format(self, layer_names):
        """
        自动检测层名格式
        """
        layer_info = {}
        other_regions = []
        
        for name in layer_names:
            found = False
            for pattern in self.layer_patterns:
                match = re.match(pattern, str(name), re.IGNORECASE)
                if match:
                    layer_num = int(match.group(1))
                    layer_info[name] = layer_num
                    found = True
                    break
            
            if not found:
                # 非层区域（如WM, GM等）
                other_regions.append(name)
        
        return layer_info, other_regions
    
    def create_standardized_mapping(self, layer_names):
        """
        创建标准化映射
        """
        layer_info, other_regions = self.detect_layer_format(layer_names)
        
        # 创建映射字典
        mapping = {}
        
        # 映射层
        for original_name, layer_num in layer_info.items():
            standard_name = f'Layer{layer_num}'
            mapping[original_name] = standard_name
        
        # 保持其他区域名称不变
        for region in other_regions:
            mapping[region] = region
        
        return mapping, layer_info, other_regions
    
    def extract_layer_numbers_from_celltype(self, celltype_name):
        """
        从细胞类型名称中提取层信息
        例如：'Ex_8_L5_6' -> [5, 6]
        """
        # 常见模式
        patterns = [
            r'L(\d+)_(\d+)',     # L5_6
            r'L(\d+)',           # L5
            r'Layer(\d+)_(\d+)', # Layer5_6  
            r'Layer(\d+)',       # Layer5
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, celltype_name)
            if matches:
                if isinstance(matches[0], tuple):
                    # 匹配到范围，如 L5_6
                    return [int(x) for x in matches[0]]
                else:
                    # 匹配到单个层
                    return [int(matches[0])]
        
        return []
    
    def auto_generate_cell_layer_mapping(self, celltype_names, available_layers):
        """
        根据细胞类型名称和可用层，自动生成细胞类型-层特异性映射
        """
        mapping = {}
        
        for celltype in celltype_names:
            if celltype.startswith('Ex_'):  # 兴奋性神经元
                layer_nums = self.extract_layer_numbers_from_celltype(celltype)
                
                if layer_nums:
                    # 将层号转换为标准格式
                    target_layers = []
                    for layer_num in layer_nums:
                        standard_layer = f'Layer{layer_num}'
                        if standard_layer in available_layers:
                            target_layers.append(standard_layer)
                    
                    # 处理范围情况（如L4_6表示L4,L5,L6）
                    if len(layer_nums) == 2:
                        start, end = min(layer_nums), max(layer_nums)
                        target_layers = []
                        for i in range(start, end + 1):
                            standard_layer = f'Layer{i}'
                            if standard_layer in available_layers:
                                target_layers.append(standard_layer)
                    
                    mapping[celltype] = target_layers
        
        return mapping

def smart_roc_analysis(adata, cluster_column='cluster'):
    """
    智能ROC分析 - 自动适配各种数据格式
    """
    print("🔍 开始智能数据格式检测...")
    
    # 1. 初始化适配器
    adapter = SmartLayerAdapter()
    
    # 2. 检测层名格式
    original_layers = adata.obs[cluster_column].unique()
    layer_mapping, layer_info, other_regions = adapter.create_standardized_mapping(original_layers)
    
    # 3. 应用标准化映射
    adata_processed = adata.copy()
    adata_processed.obs['result'] = adata_processed.obs[cluster_column].map(layer_mapping)
    
    # 4. 检测兴奋性神经元类型
    ex_celltypes = [col for col in adata.obs.columns if col.startswith('Ex_')]

    # 5. 自动生成细胞类型-层特异性映射
    available_standard_layers = [f'Layer{i}' for i in sorted(layer_info.values())]
    auto_cell_mapping = adapter.auto_generate_cell_layer_mapping(ex_celltypes, available_standard_layers)
    
    # 6. 执行ROC分析
    print(f"\n🚀 开始执行ROC分析...")
    auc_results = {}
    roc_curves = {}
    
    for celltype, target_layers in auto_cell_mapping.items():
        if not target_layers:
            print(f"⚠️  {celltype}: 跳过（无匹配层）")
            continue
        
        # 获取预测比例
        predicted_proportions = adata_processed.obs[celltype].values
        
        # 创建二分类标签
        true_layers = adata_processed.obs['result'].values
        y_true = np.array([1 if layer in target_layers else 0 for layer in true_layers])
        y_scores = predicted_proportions
        
        # 检查数据有效性
        n_positive = np.sum(y_true)
        n_total = len(y_true)
        
        if n_positive == 0:
            print(f"⚠️  {celltype}: 跳过（目标层无样本）")
            continue
        if n_positive == n_total:
            print(f"⚠️  {celltype}: 跳过（所有样本都在目标层）")
            continue
        
        # 计算ROC和AUC
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = auc(fpr, tpr)
        
        # 保存结果
        auc_results[celltype] = {
            'auc': auc_score,
            'target_layers': target_layers,
            'n_positive': n_positive,
            'n_negative': n_total - n_positive,
            'mean_prop_in_target': np.mean(predicted_proportions[y_true == 1]),
            'mean_prop_in_other': np.mean(predicted_proportions[y_true == 0])
        }
        
        roc_curves[celltype] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc_score
        }
        
        print(f"✅ {celltype:15} | AUC: {auc_score:.3f} | 目标层: {target_layers} | 正样本: {n_positive:4d}")
    
    # 7. 显示总结
    if auc_results:
        auc_scores = [result['auc'] for result in auc_results.values()]
        print(f"\n📈 分析完成!")
        print(f"   成功分析的细胞类型: {len(auc_results)}")
        print(f"   平均AUC: {np.mean(auc_scores):.3f}")
        print(f"   AUC范围: {min(auc_scores):.3f} - {max(auc_scores):.3f}")
        
        # 显示最佳和最差表现
        best_celltype = max(auc_results.keys(), key=lambda x: auc_results[x]['auc'])
        worst_celltype = min(auc_results.keys(), key=lambda x: auc_results[x]['auc'])
        print(f"   最佳表现: {best_celltype} (AUC: {auc_results[best_celltype]['auc']:.3f})")
        print(f"   最差表现: {worst_celltype} (AUC: {auc_results[worst_celltype]['auc']:.3f})")
    else:
        print(f"❌ 未找到可分析的细胞类型！")
        return None, None
    
    return auc_results, roc_curves


def plot_roc_curves(auc_results, roc_curves, figsize=(15, 6)):
    """
    Plot ROC curves (left) and AUC ranking bar plot (right)
    """

    # 一行两列子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # ========== 1. ROC 曲线 ==========
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_curves)))
    
    for i, (celltype, roc_data) in enumerate(roc_curves.items()):
        fpr, tpr, auc_score = roc_data['fpr'], roc_data['tpr'], roc_data['auc']
        
        ax1.plot(fpr, tpr, 
                color=colors[i], 
                linewidth=2.5,
                label=f'{celltype} (AUC: {auc_score:.3f})',
                alpha=0.8)
        
        if i < 3:
            ax1.fill_between(fpr, tpr, alpha=0.1, color=colors[i])
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=2, label='Random Classifier (AUC: 0.5)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax1.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ========== 2. AUC 排序条形图 ==========
    celltypes = list(auc_results.keys())
    auc_scores = [auc_results[ct]['auc'] for ct in celltypes]
    
    sorted_indices = np.argsort(auc_scores)[::-1]
    sorted_celltypes = [celltypes[i] for i in sorted_indices]
    sorted_auc_scores = [auc_scores[i] for i in sorted_indices]
    
    # 归一化分数到 0-1 映射到 colormap
    normed_scores = (np.array(sorted_auc_scores) - min(sorted_auc_scores)) / (
                     max(sorted_auc_scores) - min(sorted_auc_scores) + 1e-8)
    colors_bar = plt.cm.RdYlGn(normed_scores)
    
    bars = ax2.barh(range(len(sorted_celltypes)), sorted_auc_scores, 
                   color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    for i, (bar, score) in enumerate(zip(bars, sorted_auc_scores)):
        ax2.text(score + 0.01, i, f'{score:.3f}', 
                va='center', ha='left', fontweight='bold', fontsize=10)
    
    ax2.set_yticks(range(len(sorted_celltypes)))
    ax2.set_yticklabels(sorted_celltypes, fontsize=10)
    ax2.set_xlabel('AUC Score', fontsize=12)
    ax2.set_title('Cell Type AUC Score Ranking', fontsize=14, fontweight='bold')
    ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Random Level')
    ax2.axvline(x=np.mean(sorted_auc_scores), color='blue', linestyle='--', alpha=0.7, label='Average AUC')
    ax2.legend()
    ax2.grid(True, axis='x', alpha=0.3)
    ax2.set_xlim(0, 1)
    
    fig.suptitle('Excitatory Neuron Cell Type Layer-Specific Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    

