import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import warnings
import sys  
sys.path.append("./")
from utils.func_utils import df_desc


warnings.filterwarnings("ignore")
current_directory = os.getcwd() +'/resources'
print(current_directory)
# 读取数据
df = pd.read_csv(current_directory+'/'+'sample_cardio_good_fitness.csv')

df_desc(df)


def test_1():
    # 绘制 age 的收入柱状图
    plt.figure(figsize=(10, 5))
    # 生成合理的 y 轴刻度
    plt.bar(df['age'], df['income'], color='skyblue')
    plt.xlabel('Age')
    plt.ylabel('Income')
    plt.title('Income by Age By Sgl')
    plt.xticks(df['age'])  # 设置 x 轴刻度
    plt.yticks(df['income']) 
    # # 设置 y 轴范围
    # plt.ylim(29000, 110000)
    # # 生成合理的 y 轴刻度
    # y_ticks = np.arange(29000, 110001, 10000)  # 从29000到110000，每10000一个刻度
    # plt.yticks(y_ticks)
    plt.show()

def test_2():
    # 2. 绘制 education 和 fitness 的散点图
    plt.figure(figsize=(10, 5))
    plt.scatter(df['education'], df['fitness'], color='orange')
    plt.xlabel('Education')
    plt.ylabel('Fitness')
    plt.title('Scatter Plot of Education vs Fitness By Sgl')
    plt.grid()
    plt.show()

def test_3():
    # 3. 绘制 prodcut 和 gender 的饼图
    df['miles'] = df['miles'].astype(int)
    gender_miles = df.groupby('gender')['miles'].sum()

    # 绘制饼图
    plt.figure()
    plt.pie(gender_miles, labels= gender_miles.index, autopct='%1.1f%%', startangle=90)
    plt.title('Gender vs Miles By Sgl')
    # plt.axis('equal')  # 使饼图为圆形
    plt.show()

test_3();