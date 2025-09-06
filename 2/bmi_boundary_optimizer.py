#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
问题二专用：BMI分界值分组与NIPT时点建议（与问题三彻底分离）
- 独立实现，不依赖 3/nipt_timing_optimization.py
- 仅使用 pandas/numpy 完成分组与建议时点计算

输出字段（供问题二的 BMIBoundaryStrategy 使用）：
- BMI组别
- BMI范围_标准化
- BMI均值_标准化
- 样本数
- 建议检测时点_原始周数
"""

import pandas as pd
import numpy as np

class NIPTTimingOptimizer:
    """
    NIPT时点优化器（问题二本地实现）
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        # 标准化参数（与全项目保持一致）
        self.time_mean = 16.846
        self.time_std = 4.076
        # 预设BMI边界（标准化值）用于分组，可根据数据实际进行覆盖或优化
        self.bmi_boundaries = [-3.857, -1.410, -1.374, -1.343, -1.130,
                               0.360, 0.635, 1.085, 1.210, 2.073, 2.274, 2.935, 4.845]
        self.detection_threshold = None

    def _to_original_weeks(self, std_weeks):
        return std_weeks * self.time_std + self.time_mean

    def load_data(self) -> bool:
        try:
            self.data = pd.read_csv(self.data_path)
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False

    def assign_bmi_groups(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("数据未加载")
        df = self.data.copy()
        # 按边界生成组别（左闭右开），过滤极端组别
        groups = []
        boundaries = sorted(self.bmi_boundaries)
        for v in df['BMI_标准化']:
            idx = None
            # 区间 [-inf, b0), [b0, b1), ..., [bk-1, bk), [bk, +inf)
            for i, b in enumerate(boundaries):
                if v < b:
                    idx = i
                    break
            if idx is None:
                idx = len(boundaries)
            # 过滤最两端
            if idx == 0 or idx == len(boundaries):
                idx = np.nan
            groups.append(idx)
        df['BMI组别'] = groups
        df = df[df['BMI组别'].notna()].copy()
        df['BMI组别'] = df['BMI组别'].astype(int)
        return df

    def calculate_detection_threshold(self, df: pd.DataFrame) -> float:
        # 选择整体Y浓度10%分位作为阈值，保证约90%可检出
        thr = float(np.percentile(df['Y染色体浓度'].values, 10))
        self.detection_threshold = thr
        return thr

    def analyze_timing_by_bmi_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.detection_threshold is None:
            self.calculate_detection_threshold(df)
        out = []
        for g in sorted(df['BMI组别'].unique()):
            gdf = df[df['BMI组别'] == g]
            succ = gdf[gdf['Y染色体浓度'] >= self.detection_threshold]
            if len(succ) == 0:
                # 若该组无达标样本，则跳过
                continue
            # 采用成功样本的10%分位作为“尽早”建议时点（标准化）
            q = float(succ['孕周_标准化'].quantile(0.1))
            q_orig = float(self._to_original_weeks(q))
            out.append({
                'BMI组别': int(g),
                'BMI范围_标准化': f"[{gdf['BMI_标准化'].min():.3f}, {gdf['BMI_标准化'].max():.3f}]",
                'BMI均值_标准化': float(gdf['BMI_标准化'].mean()),
                '样本数': int(len(gdf)),
                '建议检测时点_原始周数': q_orig,
            })
        return pd.DataFrame(out).sort_values('BMI组别').reset_index(drop=True)

    def run_analysis(self) -> pd.DataFrame:
        ok = self.load_data()
        if not ok:
            return pd.DataFrame()
        df = self.assign_bmi_groups()
        recs = self.analyze_timing_by_bmi_groups(df)
        return recs
