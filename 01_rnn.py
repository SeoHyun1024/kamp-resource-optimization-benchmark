"""
01 RNN 피크전력 예측 — baseline_rnn.py 수정본 (단독 실행 파일, 다른 .py import 없음)

실행
    python 01_rnn.py                              # target = power = max(15/30/45/60분) (RF/XGB/CatBoost와 같은 정의)
    python 01_rnn.py --target 15분                # target = 15분 (기존 baseline_rnn.py·01 노트북과 같은 정의)
    python 01_rnn.py --variant D_multi_gru        # 구조 하나만 CV(2 fold) 후 최종 학습 (빠른 실행)
    python 01_rnn.py --variant D_multi_gru --skip-cv --epochs 90   # CV 없이 최종 학습만 (cutoff 167 고정)

데이터: 스크립트 폴더 기준 data/, data/legacy/, 같은 폴더 순으로 okm_augumented_2021.csv 자동 탐색
결과: results/2_test_predictions/rnn_forecast.csv   (테스트 예측)
      results/3_comparison/rnn_metrics.txt           (테스트 성능 요약 -> 보고서 수치는 여기서 확인)
      results/4_model_details/rnn_cv.csv             (구조별 CV 결과)
      results/4_model_details/rnn_oof.csv            (선택 구조의 7·8월 CV 예측 -> 09 경보 조합 선택에 사용)
      results/6_figures/rnn_forecast_plot.png

------------------------------------------------------------------------------
baseline_rnn.py 대비 수정 사항
------------------------------------------------------------------------------
[1] 전처리 추가 (원본은 원자료를 그대로 사용)
    - "시간" 이상치(2021-07-13, 07-15, 값 70~188) -> 날짜 안 행 순서로 0~23시 재구성
    - 풍속(3)·강수량(1) 결측 -> 시간 보간 / 공장인원(17) -> 생산량 0 구간 확인 후 0
    - 증강 복제일 탐지: 257일 중 115일이 이전 날의 15분 값 96개와 완전히 동일(1~7월).
      행은 지우지 않고(시계열 연속성 유지), CV 검증 구간에서만 제외한다.
[2] Early stopping: monitor="loss"(train loss) -> 별도 멈춤 판단 구간의 val_loss.
    train loss 기준이면 과적합 시점을 알 수 없어 사실상 60 epoch 고정과 같았다.
    멈춤 판단 구간은 fold마다 학습 구간의 '원본 일자 마지막 14일'로 따로 떼어 둔다.
    7·8월 검증 구간은 멈춤 판단에 쓰지 않고 채점(OOF)에만 쓴다 -> OOF가 RF OOF와 같은 조건이 된다.
[3] 검증 구간 도입: 7월·8월 확장창 2-fold CV
      fold1: 7/1 이전 학습 -> 7월 검증 / fold2: 8/1 이전 학습 -> 8월 검증
    -> 입력 구조 선택, best epoch, 피크 판정 cutoff를 여기서만 정한다.
    최종 모델은 9/1 이전 전체로 best epoch만큼 재학습, test(9/1~9/14, 336시간)는 1회만 평가.
[4] 스케일러: 학습 구간으로만 fit (원본과 동일 원칙 유지, fold마다 다시 fit)
[5] 입력 구조 후보 (CV MAE가 가장 낮은 것을 선택)
      A_original  : (168, 1) SimpleRNN(64)-Dense(32)-Dense(1)   <- baseline_rnn.py 구조 그대로
      B_weekly    : (7, 24)  SimpleRNN(50)-Dense(1)             <- 초기 01 노트북 구조
      C_multi_rnn : (168, 8) SimpleRNN(64)-Dense(32)-Dense(1)   전력·15분·60분·생산량·시간/요일 sin,cos
      D_multi_gru : (168, 8) GRU(64)-Dense(32)-Dense(1)         C와 같은 입력, GRU 셀
[6] 시드 고정 + enable_op_determinism -> 재실행 시 같은 결과
[7] target=power 일 때 "예측값 >= cutoff" 로 피크(>=179) 판정 성능(F1)도 계산
    (피크 임계값 179 = 2021-01-08~07-31 power의 90% quantile, 다른 모델과 동일한 test 라벨 47건)

2021 데이터 실행 결과 (target=power, seed 42) — 멈춤 판단 구간 분리 이전 버전 기준. 재실행하면 값이 바뀔 수 있다
    CV MAE: A 11.39 / B 14.14 / C 9.59 / D 8.24 -> D_multi_gru 선택
    Test  : MAE 6.71, RMSE 9.12 (lag_1 14.52, lag_168 8.95)
            피크 판정(예측 >= 167): Precision 0.592, Recall 0.957, F1 0.732
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # 어느 폴더에서 실행해도 스크립트 위치 기준
_CANDIDATES = [os.path.join(BASE_DIR, "data", "okm_augumented_2021.csv"),
               os.path.join(BASE_DIR, "data", "legacy", "okm_augumented_2021.csv"),
               os.path.join(BASE_DIR, "okm_augumented_2021.csv")]
DATA_PATH = next((p for p in _CANDIDATES if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError("okm_augumented_2021.csv 를 찾지 못했습니다. 다음 중 한 곳에 두세요:\n  " + "\n  ".join(_CANDIDATES))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
_OUT_DIRS = {"pred": "2_test_predictions", "cmp": "3_comparison", "detail": "4_model_details", "model": "5_models", "fig": "6_figures"}


def out(kind, name):
    d = os.path.join(RESULTS_DIR, _OUT_DIRS[kind])
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)

SEED = 42
QUARTERS = ["15분", "30분", "45분", "60분"]
N_LAGS = 168
SAMPLE_START = pd.Timestamp("2021-01-15")   # RF/XGB/CatBoost 데이터셋과 같은 예측 시작 시점
PEAK_WINDOW = (pd.Timestamp("2021-01-08"), pd.Timestamp("2021-08-01"))
PEAK_QUANTILE = 0.90
TEST_START = pd.Timestamp("2021-09-01")
CV_FOLDS = [(pd.Timestamp("2021-07-01"), pd.Timestamp("2021-08-01")),
            (pd.Timestamp("2021-08-01"), pd.Timestamp("2021-09-01"))]
PATIENCE, BATCH = 10, 128
STOP_DAYS = 14   # early stopping 판단용으로 학습 구간 끝에서 떼어 두는 원본 일자 수
CUTOFF_GRID = np.arange(165, 190, 1.0)

VARIANTS = {
    "A_original": {"channels": "uni", "shape": "168x1", "cell": "rnn", "units": 64, "dense": 32},
    "B_weekly": {"channels": "uni", "shape": "7x24", "cell": "rnn", "units": 50, "dense": None},
    "C_multi_rnn": {"channels": "multi", "shape": "168xC", "cell": "rnn", "units": 64, "dense": 32},
    "D_multi_gru": {"channels": "multi", "shape": "168xC", "cell": "gru", "units": 64, "dense": 32},
}

parser = argparse.ArgumentParser()
parser.add_argument("--target", default="power", choices=["power", "15분"])
parser.add_argument("--variant", default=None, choices=list(VARIANTS), help="지정하면 이 구조만 CV 후 학습")
parser.add_argument("--skip-cv", action="store_true", help="--variant와 함께: CV 없이 최종 학습만 (OOF 미생성)")
parser.add_argument("--epochs", type=int, default=90, help="--skip-cv 사용 시 최종 학습 epoch")
parser.add_argument("--max-epochs", type=int, default=150, help="CV early stopping 최대 epoch")
args = parser.parse_args()
TARGET = args.target
TAG = "rnn" if TARGET == "power" else "rnn_15min"

keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()


# =============================================================================
# 단계1~3: 데이터 불러오기 / 확인 / 정제
# =============================================================================
df = pd.read_csv(DATA_PATH)
print(f"[정보] 데이터: {DATA_PATH}  shape: {df.shape}")

assert (df.groupby("날짜").size() == 24).all(), "하루 24행이 아닌 날짜가 있음"
broken = df.loc[~df["시간"].between(0, 23), "날짜"].unique().tolist()
df["시간"] = df.groupby("날짜").cumcount()
ts = pd.to_datetime(df["날짜"].astype(str), format="%Y%m%d") + pd.to_timedelta(df["시간"], unit="h")
df = df.set_index(pd.DatetimeIndex(ts, name="Date")).sort_index()
assert (df.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all(), "1시간 간격 아님"
print(f"[정보] '시간' 이상치 날짜 {broken} -> 날짜 내 행 순서로 0~23시 재구성")

na = df.isna().sum()
df["풍속"] = df["풍속"].interpolate(method="time", limit_direction="both")
df["강수량"] = df["강수량"].interpolate(method="time", limit_direction="both")
assert (df.loc[df["공장인원"].isna(), "생산량"] == 0).all()
df["공장인원"] = df["공장인원"].fillna(0)
print(f"[정보] 결측 {na[na > 0].to_dict()} -> 풍속·강수량 보간, 공장인원 0")

df["power"] = df[QUARTERS].max(axis=1).astype(float)

day_key = df.groupby("날짜")[QUARTERS].apply(lambda g: tuple(g.to_numpy().ravel()))
seen, copied = set(), set()
for d, key in day_key.items():
    (copied.add(d) if key in seen else seen.add(key))
df["is_copied_day"] = df["날짜"].isin(copied).astype(int)
print(f"[정보] 복제일 {len(copied)}일 / {len(day_key)}일 (CV 검증 구간에서 제외)")

peak_thr = float(df.loc[(df.index >= PEAK_WINDOW[0]) & (df.index < PEAK_WINDOW[1]), "power"].quantile(PEAK_QUANTILE))
df["label"] = (df["power"] >= peak_thr).astype(int)
print(f"[정보] 피크 임계값 {peak_thr:.0f} (target=power일 때 피크 판정에 사용)")

y_all = df[TARGET].astype(float).to_numpy()
hour, dow = df.index.hour.to_numpy(), df.index.dayofweek.to_numpy()
CHANNELS = {
    "uni": np.c_[df[TARGET]].astype(float),
    "multi": np.c_[
        df["power"], df["15분"], df["60분"], df["생산량"],
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
    ].astype(float),
}

samples = df.loc[df.index >= SAMPLE_START]


def rows_of(frame):
    return df.index.get_indexer(frame.index)


def cv_folds():
    out = []
    for start, end in CV_FOLDS:
        va = samples.loc[(samples.index >= start) & (samples.index < end)]
        out.append((start.strftime("%Y-%m"), samples.loc[samples.index < start], va.loc[va["is_copied_day"] == 0]))
    return out


def stop_split(tr):
    """학습 구간을 (내부 학습, 멈춤 판단)으로 나눈다. 멈춤 판단 = 복제일이 아닌 마지막 STOP_DAYS일.
    내부 학습은 멈춤 판단 구간 시작 이전 행만 써서 시간 순서를 지킨다."""
    days = tr.index.normalize()
    orig_days = np.sort(days[tr["is_copied_day"].to_numpy() == 0].unique())
    stop_days = orig_days[-STOP_DAYS:]
    stop = tr.loc[days.isin(stop_days)]
    inner = tr.loc[tr.index < stop.index.min()]
    return inner, stop


# =============================================================================
# 단계4: 입력 시퀀스 / 모델
# =============================================================================
def make_xy(cfg, rows, fit_rows):
    """rows: 예측 시점 T의 행 번호 -> 입력은 T-168 ~ T-1. 스케일러는 학습 구간(fit_rows)까지로만 fit."""
    raw = CHANNELS[cfg["channels"]]
    sx = MinMaxScaler().fit(raw[: fit_rows.max() + 1])
    sy = MinMaxScaler().fit(y_all[fit_rows].reshape(-1, 1))
    scaled = sx.transform(raw)
    X = np.stack([scaled[r - N_LAGS: r] for r in rows])
    if cfg["shape"] == "7x24":
        X = X[:, :, 0].reshape(-1, 7, 24)   # 과거->최신 순 (7일, 24시간)
    make_xy.last_sx = sx  # 최종 모델 저장 시 입력 스케일러도 함께 남기기 위해
    return X, sy.transform(y_all[rows].reshape(-1, 1)).ravel(), sy


def build(cfg, input_shape):
    cell = keras.layers.SimpleRNN if cfg["cell"] == "rnn" else keras.layers.GRU
    layers = [keras.layers.Input(shape=input_shape), cell(cfg["units"])]
    if cfg["dense"]:
        layers.append(keras.layers.Dense(cfg["dense"], activation="relu"))
    layers.append(keras.layers.Dense(1))
    model = keras.Sequential(layers)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def peak_scores(y_true, pred, cutoff):
    p = (np.asarray(pred) >= cutoff).astype(int)
    return {"Precision": precision_score(y_true, p, zero_division=0),
            "Recall": recall_score(y_true, p, zero_division=0),
            "F1": f1_score(y_true, p, zero_division=0)}


# =============================================================================
# 단계5: CV로 구조·epoch·cutoff 선택
# =============================================================================
def run_cv(name, cfg):
    """2-fold CV. 멈춤 판단 구간으로 epoch를 정하고, 7·8월은 예측(OOF)에만 쓴다."""
    preds, trues, labels, dates, folds_used, epochs = [], [], [], [], [], []
    for fold_name, tr, va in cv_folds():
        keras.utils.set_random_seed(SEED)
        inner, stop = stop_split(tr)
        in_rows, st_rows, va_rows = rows_of(inner), rows_of(stop), rows_of(va)
        Xin, yin, sy = make_xy(cfg, in_rows, in_rows)
        Xst, yst, _ = make_xy(cfg, st_rows, in_rows)
        Xva, _, _ = make_xy(cfg, va_rows, in_rows)
        model = build(cfg, Xin.shape[1:])
        hist = model.fit(Xin, yin, validation_data=(Xst, yst), epochs=args.max_epochs, batch_size=BATCH,
                         verbose=0, callbacks=[keras.callbacks.EarlyStopping(
                             monitor="val_loss", patience=PATIENCE, restore_best_weights=True)])
        epochs.append(int(np.argmin(hist.history["val_loss"])) + 1)
        preds.append(sy.inverse_transform(model.predict(Xva, verbose=0)).ravel())
        trues.append(y_all[va_rows]); labels.append(va["label"].to_numpy())
        dates.append(va.index); folds_used.append(np.repeat(fold_name, len(va)))
        print(f"    {name} fold {fold_name}: 멈춤 판단 {stop.index.normalize().min():%m-%d}~{stop.index.normalize().max():%m-%d} "
              f"(원본 {STOP_DAYS}일), 내부 학습 {len(inner)}행, best epoch {epochs[-1]}")
    p, t, lab = map(np.concatenate, (preds, trues, labels))
    row = {"variant": name, "cv_MAE": mean_absolute_error(t, p), "cv_RMSE": float(np.sqrt(mean_squared_error(t, p))),
           "best_epoch": int(np.mean(epochs)), "fold_epochs": epochs}
    oof_frame = pd.DataFrame({"Date": np.concatenate(dates), "fold": np.concatenate(folds_used),
                              "label": lab, "actual": t, "forecast": p})
    return row, oof_frame


if args.variant and args.skip_cv:
    best_name, n_epochs, cutoff = args.variant, args.epochs, 167.0
    print(f"[정보] CV 생략: {best_name}, epoch {n_epochs}, cutoff {cutoff:.0f}(기본값). OOF는 만들지 않음")
else:
    targets = {args.variant: VARIANTS[args.variant]} if args.variant else VARIANTS
    cv_rows, oof = [], {}
    for name, cfg in targets.items():
        row, oof[name] = run_cv(name, cfg)
        cv_rows.append(row)
        print(f"[CV] {name:<12} MAE {row['cv_MAE']:.3f} | RMSE {row['cv_RMSE']:.3f} | best epoch {row['fold_epochs']}")

    cv = pd.DataFrame(cv_rows).sort_values("cv_MAE")
    if not args.variant:  # 구조 비교표는 전체 실행일 때만 덮어쓴다
        cv.to_csv(out("detail", f"{TAG}_cv.csv"), index=False)
    best_name, n_epochs = cv.iloc[0]["variant"], int(cv.iloc[0]["best_epoch"])
    oof_best = oof[best_name]
    if TARGET == "power":
        f1s = [peak_scores(oof_best["label"], oof_best["forecast"], c)["F1"] for c in CUTOFF_GRID]
        cutoff = float(CUTOFF_GRID[int(np.argmax(f1s))])
        oof_best = oof_best.assign(predicted_label=(oof_best["forecast"] >= cutoff).astype(int))
    else:
        cutoff = 167.0
    oof_best.to_csv(out("detail", f"{TAG}_oof.csv"), index=False)
    print(f"[선택] {best_name} (CV MAE {cv.iloc[0]['cv_MAE']:.3f}), 최종 epoch {n_epochs}, 피크 cutoff {cutoff:.0f}"
          f" (7·8월 OOF F1 최대) -> results/4_model_details/{TAG}_oof.csv 저장")

# =============================================================================
# 단계6: 최종 학습(9/1 이전 전체) + test 1회 평가
# =============================================================================
cfg = VARIANTS[best_name]
train_part, test_part = samples.loc[samples.index < TEST_START], samples.loc[samples.index >= TEST_START]
keras.utils.set_random_seed(SEED)
tr_rows, te_rows = rows_of(train_part), rows_of(test_part)
Xtr, ytr, sy = make_xy(cfg, tr_rows, tr_rows)
sx_final = make_xy.last_sx
Xte, _, _ = make_xy(cfg, te_rows, tr_rows)
model = build(cfg, Xtr.shape[1:])
model.fit(Xtr, ytr, epochs=n_epochs, batch_size=BATCH, verbose=0)
pred = sy.inverse_transform(model.predict(Xte, verbose=0)).ravel()
actual = y_all[te_rows]

rows = {
    "Persistence (lag_1)": y_all[te_rows - 1],
    "1주 전 같은 시각 (lag_168)": y_all[te_rows - 168],
    f"RNN {best_name}": pred,
}
table = pd.DataFrame({name: {"MAE": mean_absolute_error(actual, p),
                             "RMSE": float(np.sqrt(mean_squared_error(actual, p)))} for name, p in rows.items()}).T
print(f"\n=== Test (2021-09-01~09-14, {len(actual)}시간), target = {TARGET} ===")
print(table.round(3).to_string())
if TARGET == "15분":
    print("참고: 기존 01 노트북 SimpleRNN MAE 7.872 / RMSE 12.085")

lines = [f"target={TARGET}", f"variant={best_name}", f"epochs={n_epochs}",
         f"test_mae={table.iloc[-1]['MAE']:.4f}", f"test_rmse={table.iloc[-1]['RMSE']:.4f}"]
pred_table = pd.DataFrame({"Date": test_part.index, "actual": actual, "forecast": pred})
if TARGET == "power":
    s = peak_scores(test_part["label"].to_numpy(), pred, cutoff)
    print(f"\n피크 판정(예측 >= {cutoff:.0f}): Precision {s['Precision']:.4f} | Recall {s['Recall']:.4f} | F1 {s['F1']:.4f}"
          f"  (test 피크 {int(test_part['label'].sum())}건)")
    lines += [f"peak_cutoff={cutoff:.0f}", f"peak_precision={s['Precision']:.4f}",
              f"peak_recall={s['Recall']:.4f}", f"peak_f1={s['F1']:.4f}"]
    pred_table["actual_label"] = test_part["label"].to_numpy()
    pred_table["predicted_label"] = (pred >= cutoff).astype(int)

pred_table.to_csv(out("pred", f"{TAG}_forecast.csv"), index=False)
with open(out("cmp", f"{TAG}_metrics.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\n[정보] 저장: results/2_test_predictions/{TAG}_forecast.csv, results/3_comparison/{TAG}_metrics.txt")

# 최종 모델 + 스케일러 + 메타데이터 저장 (results/5_models/) ---------------------------------
import json
import platform
import joblib

model.save(out("model", f"{TAG}.keras"))
joblib.dump({"x_scaler": sx_final, "y_scaler": sy}, out("model", f"{TAG}_scalers.joblib"))
channel_names = ([TARGET] if cfg["channels"] == "uni" else
                 ["power", "15분", "60분", "생산량", "hour_sin", "hour_cos", "dow_sin", "dow_cos"])
meta = {
    "model": TAG, "model_file": f"{TAG}.keras", "scaler_file": f"{TAG}_scalers.joblib",
    "script": "01_rnn.py", "task": f"다음 시간 {TARGET} 예측" + (" + cutoff로 피크 판정" if TARGET == "power" else ""),
    "variant": best_name, "architecture": {k: v for k, v in cfg.items()},
    "input": f"T-{N_LAGS} ~ T-1 시퀀스, 채널 = {channel_names}",
    "input_shape": list(Xtr.shape[1:]), "channels": channel_names,
    "scaling": "MinMaxScaler (입력 채널별 x_scaler, target y_scaler), 학습 구간(9/1 이전)으로만 fit",
    "epochs": n_epochs, "batch_size": BATCH, "optimizer": "Adam(lr=1e-3)", "loss": "mse",
    "early_stopping": (f"CV fold마다 학습 구간의 원본 일자 마지막 {STOP_DAYS}일로 판단 (7·8월 검증 구간은 채점에만 사용)"
                       if not (args.variant and args.skip_cv) else "CV 생략 (--skip-cv), epoch 고정"),
    "cutoff_selection": "7·8월 OOF F1 최대" if not (args.variant and args.skip_cv) else "기본값 167",
    "peak_threshold": peak_thr,
    "train_period": [str(train_part.index.min()), str(train_part.index.max())],
    "test_period": [str(test_part.index.min()), str(test_part.index.max())],
    "seed": SEED, "python": platform.python_version(), "tensorflow": tf.__version__,
    "test": {"MAE": float(table.iloc[-1]["MAE"]), "RMSE": float(table.iloc[-1]["RMSE"])},
}
if TARGET == "power":
    meta.update({"decision_rule": f"predict >= {cutoff}", "peak_cutoff": cutoff,
                 "test": {**meta["test"], **{k: round(float(v), 4) for k, v in s.items()}}})
with open(out("model", f"{TAG}_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
print(f"[정보] 모델 저장: results/5_models/{TAG}.keras + {TAG}_scalers.joblib + {TAG}_meta.json")

# 시각화 --------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
_installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR",
                                                 "Noto Sans CJK JP", "Noto Sans CJK TC"] if f in _installed), "DejaVu Sans")
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(11, 4))
plt.plot(test_part.index, actual, label=f"실측 ({TARGET})", linewidth=1.2)
plt.plot(test_part.index, pred, label=f"RNN 예측 ({best_name})", linewidth=1.2, alpha=0.85)
if TARGET == "power":
    plt.axhline(peak_thr, color="red", linestyle="--", linewidth=0.8, label=f"피크 임계값 {peak_thr:.0f}")
plt.title("RNN 피크전력 예측 — 2021-09-01~09-14 테스트 구간")
plt.legend()
plt.tight_layout()
plt.savefig(out("fig", f"{TAG}_forecast_plot.png"), dpi=130)