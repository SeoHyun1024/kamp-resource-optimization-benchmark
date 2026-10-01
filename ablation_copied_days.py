"""
보조 실험: 복제일(증강으로 복사된 115일)을 학습에서 어떻게 다룰지 CV로 결정

비교: 유지 / 가중치 0.5 / 요일이 원본과 다른 복제일만 제외 / 전부 제외
평가: 7·8월 확장창 CV OOF PR-AUC, F1 (XGBoost 고정 파라미터, seed 3개 평균)
결과(2021 데이터 기준): 유지 0.861 / w0.5 0.860 / 요일불일치 제외 0.852 / 전부 제외 0.843
 -> preprocessing.DROP_COPIED_FROM_TRAIN = False (유지)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

import common as cm
import preprocessing as pp

pp.DROP_COPIED_FROM_TRAIN = False  # 여기서 직접 가중치로 제어
ds, F, _ = pp.load_dataset(log=lambda *_: None)
copy_of = pp.clean(pp.load_raw(), log=lambda *_: None).attrs["copy_of"]
dow = lambda d: pd.Timestamp(str(d)).dayofweek
inconsistent = {d for d, s in copy_of.items() if dow(d) != dow(s)}
print(f"[정보] 복제일 {len(copy_of)}일 중 원본과 요일이 다른 날 {len(inconsistent)}일")
ds["inconsistent"] = pd.Series(ds.index.strftime("%Y%m%d").astype(int), index=ds.index).isin(inconsistent).astype(int)

for mode in ["keep", "weight_0.5", "drop_inconsistent", "drop_all"]:
    scores = []
    for seed in [42, 7, 123]:
        ys, ss = [], []
        for _, tr, va in cm.cv_folds(ds):
            w = np.ones(len(tr))
            if mode == "weight_0.5":
                w[tr["is_copied_day"].to_numpy() == 1] = 0.5
            elif mode == "drop_inconsistent":
                w[tr["inconsistent"].to_numpy() == 1] = 0
            elif mode == "drop_all":
                w[tr["is_copied_day"].to_numpy() == 1] = 0
            keep = w > 0
            t2 = tr.loc[keep]
            model = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                                  colsample_bytree=0.8, scale_pos_weight=(t2["label"] == 0).sum() / t2["label"].sum(),
                                  random_state=seed, n_jobs=-1)
            model.fit(t2[F], t2["label"], sample_weight=w[keep])
            ys.append(va["label"].to_numpy()); ss.append(model.predict_proba(va[F])[:, 1])
        y, s = np.concatenate(ys), np.concatenate(ss)
        scores.append((average_precision_score(y, s), cm.evaluate(y, s, cm.best_threshold(y, s))["F1"]))
    a = np.array(scores)
    print(f"{mode:<18} CV PR-AUC {a[:, 0].mean():.4f} ± {a[:, 0].std():.4f} | CV F1 {a[:, 1].mean():.4f} ± {a[:, 1].std():.4f}")
