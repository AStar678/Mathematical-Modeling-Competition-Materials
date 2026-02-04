# -*- coding: utf-8 -*-
"""
2026 MCM Problem C: Data With The Stars
Module: Advanced Data Preprocessing & Feature Engineering
Target: O-Award Level Analysis

优化内容：
1. 数据重构：Wide to Long 格式转换，便于时序分析。
2. 特征工程：增加Z-Score（相对竞争力）、争议度（方差）、标准化分数。
3. 异常处理：动态识别每周评委人数，修正缺省值。
"""

import pandas as pd
import numpy as np
import re
import os

# 常量配置
DATA_PATH = 'dataset/2026_MCM_Problem_C_Data.csv'
MAX_WEEKS = 11
MAX_JUDGES = 4

def load_and_clean_data(filepath=DATA_PATH):
    """
    加载原始数据并进行基础清洗
    """
    if not os.path.exists(filepath):
        # 如果找不到文件，尝试在当前目录下查找或报错
        print(f"Warning: File not found at {filepath}, checking current directory...")
        if os.path.exists(os.path.basename(filepath)):
            filepath = os.path.basename(filepath)
        else:
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
    df = pd.read_csv(filepath)
    print(f"原始数据加载成功: {df.shape[0]} 名选手, {df['season'].nunique()} 个赛季")
    
    # 1. 填补缺失的元数据
    df['celebrity_homestate'] = df['celebrity_homestate'].fillna('Unknown')
    
    # 2. 解析淘汰周次 (Parsing Elimination Week)
    # 'results' 列包含 "Eliminated Week 3", "1st Place" 等信息
    def parse_elimination(res):
        res = str(res).lower()
        if 'eliminated week' in res:
            try:
                return int(re.search(r'week (\d+)', res).group(1))
            except:
                return MAX_WEEKS + 1 
        elif any(x in res for x in ['place', 'winner', 'runner', 'finalist']):
            return MAX_WEEKS + 1 # 幸存至决赛
        elif 'withdrew' in res:
             # 尝试提取周次，若无则设为0
             match = re.search(r'week (\d+)', res)
             if match:
                 return int(match.group(1))
             return 0
        return MAX_WEEKS + 1

    df['eliminated_week'] = df['results'].apply(parse_elimination)
    return df

def process_features(df):
    """
    核心处理函数：生成宽表（Summary）和长表（Time-Series）
    并计算创新特征指标。
    """
    long_data = []
    
    for idx, row in df.iterrows():
        season = row['season']
        contestant = row['celebrity_name']
        elim_week = row['eliminated_week']
        
        for week in range(1, MAX_WEEKS + 1):
            # 提取当周评委分数
            scores = []
            judge_cols = [f'week{week}_judge{j}_score' for j in range(1, MAX_JUDGES + 1)]
            
            for col in judge_cols:
                val = row.get(col, np.nan)
                # 过滤无效分数：NaN 或 0 (假设0分代表未参赛，除非是极其罕见的真实低分)
                # 注意：有些数据中0.0代表缺席，需要结合是否已被淘汰判断
                if pd.notna(val) and val > 0:
                    scores.append(val)
            
            # 判断本周是否参赛
            # 逻辑：如果有分数，或者（当前周 < 淘汰周 且 不是退赛）
            # 这里我们主要依赖是否有分数为准，辅以淘汰周次校验
            if scores:
                week_total = sum(scores)
                judge_count = len(scores)
                
                # 计算争议指数 (Controversy Index): 评委分数的标准差
                # std=0 表示意见一致，std大表示有分歧
                judge_std = np.std(scores) if judge_count > 1 else 0
                
                record = {
                    'season': season,
                    'contestant': contestant,
                    'week': week,
                    'total_score': week_total,
                    'judge_count': judge_count,
                    'avg_score': np.mean(scores),
                    'judge_controversy': judge_std, # 创新特征：评委分歧度
                    'is_eliminated': (week == elim_week), # 本周是否被淘汰
                    'status': 'Eliminated' if week == elim_week else 'Safe',
                    'final_placement': row['placement']
                }
                
                # 保留原始评委分，用于后续特定评委分析
                for i, score in enumerate(scores):
                    record[f'judge{i+1}'] = score
                    
                long_data.append(record)

    df_long = pd.DataFrame(long_data)
    
    # --- 进阶特征工程 (Advanced Feature Engineering) ---
    
    # 1. 计算理论最高分 (用于归一化)
    # 假设单人满分10分（如有特殊赛季需调整）
    df_long['max_possible'] = df_long['judge_count'] * 10
    df_long['score_ratio'] = df_long['total_score'] / df_long['max_possible']
    
    # 2. 相对竞争力 Z-Score (Innovation!)
    # 计算公式：(选手分数 - 当周赛季平均分) / 当周赛季标准差
    # 这能反映选手在当期环境下的统治力，排除"分数通胀"的影响
    grouped = df_long.groupby(['season', 'week'])['total_score']
    df_long['week_mean'] = grouped.transform('mean')
    df_long['week_std_all'] = grouped.transform('std').fillna(1) # 防止除以0
    
    df_long['performance_z_score'] = (df_long['total_score'] - df_long['week_mean']) / df_long['week_std_all']
    
    # 3. 排名特征
    df_long['week_rank'] = df_long.groupby(['season', 'week'])['total_score'].rank(ascending=False, method='min')
    
    # 4. 动量 (Momentum): 与上周相比的排名/分数变化
    # 需要先排序
    df_long = df_long.sort_values(['season', 'contestant', 'week'])
    df_long['prev_score_ratio'] = df_long.groupby(['season', 'contestant'])['score_ratio'].shift(1)
    df_long['momentum'] = df_long['score_ratio'] - df_long['prev_score_ratio']
    
    return df, df_long

def main():
    print("Starting Advanced Data Preprocessing...")
    
    # Load
    df = load_and_clean_data()
    
    # Process
    df_wide, df_long = process_features(df)
    
    # Save
    df_long.to_csv('processed_data_long.csv', index=False)
    # df_wide 可以选择性保存，如果还需要宽表格式
    # df_wide.to_csv('processed_data_wide.csv', index=False)
    
    print("\n处理完成!")
    print(f"生成的长表维度: {df_long.shape}")
    print("\n数据预览 (Top 5):")
    print(df_long[['season', 'contestant', 'week', 'total_score', 'performance_z_score', 'judge_controversy']].head())
    
    # 简单的统计检查
    print("\n特征统计描述:")
    print(df_long[['performance_z_score', 'judge_controversy']].describe())

if __name__ == "__main__":
    main()