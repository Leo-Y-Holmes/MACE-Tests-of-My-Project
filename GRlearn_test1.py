
import numpy as np
from sklearn.utils import check_random_state
from gplearn.genetic import SymbolicRegressor
# 创建网格化训练数据
x0_range = np.arange(-1, 1, 0.1)
x1_range = np.arange(-1, 1, 0.1)
x0_grid, x1_grid = np.meshgrid(x0_range, x1_range)
true_function = x0_grid**2 - x1_grid**2 + x1_grid - 1
 
# 生成随机训练样本
rng = check_random_state(0)
training_features = rng.uniform(-1, 1, 100).reshape(50, 2)
training_target = training_features[:, 0]**2 - training_features[:, 1]**2 + training_features[:, 1] - 1


#模型配置
#1.实例化
symbolic_model = SymbolicRegressor(
    population_size=5000,      # 种群规模：越大探索越充分
    generations=20,           # 进化代数：配合停止条件使用
    stopping_criteria=0.01,   # 提前停止：避免过度进化
    p_crossover=0.7,         # 交叉概率：维持多样性
    p_subtree_mutation=0.1,  # 子树变异：引入新结构
    p_hoist_mutation=0.05,   # 提升变异：简化表达式
    p_point_mutation=0.1,      # 点变异：微调优化
    max_samples=0.9,          # 采样比例：防止过拟合
    parsimony_coefficient=0.01,  # 简洁系数：平衡复杂度
    random_state=0
)
#2.fit
symbolic_model.fit(training_features, training_target)
#3.执行
y_pred=symbolic_model._program
print(y_pred)

# 绘制二叉树图像
import graphviz
dot_data = symbolic_model._program.export_graphviz()
graph = graphviz.Source(dot_data)
# 生成 SVG（矢量图，浏览器可打开），并自动打开
graph.render('test_tree', format='svg', cleanup=True)
import os
os.startfile(os.path.abspath('test_tree.svg'))  # 用浏览器打开 SVG

