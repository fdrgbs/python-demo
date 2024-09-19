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

current_directory = os.getcwd()+'/resources'
print(current_directory)
# 读取数据
features = pd.read_csv(current_directory+'/'+'feature_data.csv')
labels = pd.read_csv(current_directory+'/'+'labels.csv')

df_desc(features)
df_desc(labels)

def test_1():
    # 选择特征列
    all_features = [col for col in features.columns if
                    col not in ["sampleid", "dataset_tag", "maxdate", "mindate"] 
                    and not col.startswith("perc_")]

    # 合并数据
    train_testdata = labels.merge(features, on='sampleid', how='left')
    train_testdata['payout_severity'].fillna(np.random.choice([0, 1]), inplace=True)
    train_testdata['has_payout'] = (train_testdata['payout_severity'] > 0.5).astype(int)

    # 填充NaN值
    for feature in all_features:
        if train_testdata[feature].dtype == 'float64':
            train_testdata[feature].fillna(train_testdata[feature].median(), inplace=True)
        else:
            train_testdata[feature].fillna(train_testdata[feature].mode()[0], inplace=True)

    train_testdata['has_payout'].fillna(np.random.choice([0, 1]), inplace=True)

    # 检查NaN值
    nan_counts = train_testdata.isnull().sum()
    # print(nan_counts)

    # 分割数据集
    train_data = train_testdata[train_testdata['dataset_tag'] == 'train']
    testdata = train_testdata[train_testdata['dataset_tag'] == 'test']

    # 确保有足够的样本
    if len(train_data) < 1 or len(testdata) < 1:
        raise ValueError("Training or testing data has insufficient samples.")

    # 训练逻辑回归模型
    X_train = train_data[all_features].values
    y_train = train_data['has_payout'].values
    model = LogisticRegressionCV()
    model.fit(X_train, y_train)

    # 预测
    X_test = testdata[all_features].values
    testdata['pred'] = model.predict_proba(X_test)[:, 1]
    train_pred_prob = model.predict_proba(X_train)[:, 1]

    # 输出预测结果
    print(testdata[['sampleid', 'pred']])

    # 计算AUC
    auc_score = roc_auc_score(y_train, train_pred_prob)
    print(f"AUC: {auc_score:.6f}")

    # 计算Spearman相关系数
    spearman_corr, _ = spearmanr(y_train, train_pred_prob)
    print(f"Spearman Correlation Coefficient: {spearman_corr:.2f}")

    # 画ROC曲线
    fpr, tpr, thresholds = roc_curve(y_train, train_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.6f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    merged_df = labels.merge(testdata[['sampleid', 'pred']], on='sampleid', how='left')
    merged_df.fillna(auc_score * 0.9 + spearman_corr * (np.random.choice([0, 1])), inplace=True)
    pd.DataFrame(merged_df).to_csv('merged_submission.csv', index=False)