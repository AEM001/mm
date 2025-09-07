import pandas as pd
import numpy as np

class NIPTTimingOptimizer:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        self.time_mean = 16.846
        self.time_std = 4.076
        self.bmi_boundaries = [-3.857, -1.410, -1.374, -1.343, -1.130,
                               0.360, 0.635, 1.085, 1.210, 2.073, 2.274, 2.935, 4.845]
        self.detection_threshold = None

    def _to_original_weeks(self, std_weeks):
        return std_weeks * self.time_std + self.time_mean

    def load_data(self) -> bool:
        self.data = pd.read_csv(self.data_path)
        return True

    def assign_bmi_groups(self) -> pd.DataFrame:
        df = self.data.copy()
        groups = []
        boundaries = sorted(self.bmi_boundaries)
        for v in df['BMI_标准化']:
            idx = None
            for i, b in enumerate(boundaries):
                if v < b:
                    idx = i
                    break
            if idx is None:
                idx = len(boundaries)
            if idx == 0 or idx == len(boundaries):
                idx = np.nan
            groups.append(idx)
        df['BMI组别'] = groups
        df = df[df['BMI组别'].notna()].copy()
        df['BMI组别'] = df['BMI组别'].astype(int)
        return df

    def calculate_detection_threshold(self, df: pd.DataFrame) -> float:
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
                continue
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
