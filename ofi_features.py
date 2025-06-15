"""
OFI feature builder – best, multi-level, integrated, and (if present) cross-asset series.

Writes
    ofi_best.csv         · best-level OFI
    ofi_multi.csv        · 10-level OFI matrix
    ofi_integrated.csv   · PCA single series
    ofi_cross_asset.csv  · only when a 'symbol' column exists
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

# load raw book 
DATA_FILE = Path(r"C:\\Users\\wadol\\Downloads\\first_25000_rows.csv")
df = pd.read_csv(DATA_FILE)

DEPTH = 10
BID_PX = [f"bid_px_{i:02d}" for i in range(DEPTH)]
ASK_PX = [f"ask_px_{i:02d}" for i in range(DEPTH)]
BID_SZ = [f"bid_sz_{i:02d}" for i in range(DEPTH)]
ASK_SZ = [f"ask_sz_{i:02d}" for i in range(DEPTH)]

# helper 
def ofi_level(book: pd.DataFrame, k: int) -> np.ndarray:
    bid, ask = BID_SZ[k], ASK_SZ[k]
    pxb, pxa = BID_PX[k], ASK_PX[k]

    d_bid_sz = book[bid].diff().fillna(0)
    d_ask_sz = book[ask].diff().fillna(0)
    d_bid_px = book[pxb].diff().fillna(0)
    d_ask_px = book[pxa].diff().fillna(0)

    buy  = np.where((d_bid_px > 0) | ((d_bid_px == 0) & (d_bid_sz > 0)),  d_bid_sz, 0)
    sell = np.where((d_ask_px < 0) | ((d_ask_px == 0) & (d_ask_sz > 0)), d_ask_sz, 0)
    return buy - sell   # +ve → buy pressure

# event-level OFI 
event_ofi = pd.DataFrame({f"ofi_lvl{k}": ofi_level(df, k) for k in range(DEPTH)},
                         index=df.index)

# resample to 1-second bars 
event_ofi.index = pd.to_datetime(df["ts_event"]).dt.tz_localize(None)
ofi_1s = event_ofi.resample("1s").sum().dropna()

# integrated OFI (PCA) 
pca = PCA(n_components=1, random_state=0)
ofi_1s["ofi_integrated"] = pca.fit_transform(ofi_1s.iloc[:, :DEPTH]).ravel()

weights = pca.components_[0] / np.abs(pca.components_[0]).sum()
print("PCA weights by depth:")
for k, w in enumerate(weights):
    print(f"  level {k}: {w:+.4f}")

# cross-asset 
if "symbol" in df.columns:
    mats = []
    for sym, g in df.groupby("symbol"):
        m = pd.DataFrame({f"{sym}_lvl{k}": ofi_level(g, k) for k in range(DEPTH)},
                         index=g.index)
        m.index = pd.to_datetime(g["ts_event"]).dt.tz_localize(None)
        mats.append(m.resample("1s").sum().dropna())
    pd.concat(mats, axis=1).sort_index().to_csv("ofi_cross_asset.csv")

# write outputs
ofi_1s["ofi_lvl0"].to_csv("ofi_best.csv")
ofi_1s.iloc[:, :DEPTH].to_csv("ofi_multi.csv")
ofi_1s[["ofi_integrated"]].to_csv("ofi_integrated.csv")

print("\nFiles written:")
files = ["ofi_best.csv", "ofi_multi.csv", "ofi_integrated.csv"]
if "symbol" in df.columns:
    files.append("ofi_cross_asset.csv")
for f in files:
    print(" ", f)
