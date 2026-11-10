# ======================================================================================================================
# RECONSTRUCT SCENARIO GROUPS FROM THE EXISTING PHASE 1 TRAIN / TEST CSVs  (no generator state, no EPANET, no scenario_ids)
# Run this anywhere (it only reads the two CSVs). Writes NEW files into the Phase 2 folder; the Phase 1 CSVs are not modified.
#
#  * Leak scenarios (>= 1 leak): EXACT. leak_C is constant within a scenario and continuous, so the leak vector is a unique fingerprint.
#  * Zero-leak scenarios: NO label fingerprint exists. They are reconstructed from scenario-level "signature" columns (operating conditions that persist
#    across the 24 hours), one row per hour per scenario enforced by an assignment solver, and then VALIDATED (see the report). If validation fails,
#    ZERO_POLICY="auto" drops the zero-leak rows instead of guessing - that is provably free of train/test scenario leakage.
# Needs: numpy, pandas, scipy.
# ======================================================================================================================
import os, json, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# ---- SET THESE ------------------------------------------------------------------------------------------------------
TRAIN_CSV = r"C:\Data\pipeline-leakage-system\Phase 1\pipeline_leak_dataset_expanded(train).csv"
TEST_CSV = r"C:\Data\pipeline-leakage-system\Phase 1\pipeline_leak_dataset_expanded(test).csv"
OUT_DIR_P2  = r"C:\Data\pipeline-leakage-system\Phase 2"
ZERO_POLICY = "auto"    # "auto": reconstruct only if validated, else drop | "reconstruct": force (warns if unvalidated) | "drop": always drop
# ---- validation thresholds (all must hold for the zero-leak reconstruction to be called RELIABLE) ---------------------
ICC_MIN             = 0.5     # a column is a scenario "signature" column if >= 50% of its (hour-adjusted) variance is between scenarios
MIN_SIGNATURE_COLS  = 5
MIN_EXACT_RECOVERY  = 0.99    # on HELD-OUT leak scenarios whose labels are hidden, >= 99% of scenarios must be recovered exactly
MAX_AMBIGUOUS_FRAC  = 0.005   # zero-leak rows whose own cluster is not clearly closer than the best other cluster
AMBIG_RATIO         = 0.5     # ambiguous if cost(own) / cost(best other) > 0.5
SEED = 0
# ---------------------------------------------------------------------------------------------------------------------
assert TRAIN_CSV and TEST_CSV, "Set TRAIN_CSV and TEST_CSV above."
assert ZERO_POLICY in ("auto", "reconstruct", "drop")
rng = np.random.default_rng(SEED); t_start = time.time()

# ---------------------------------------------- 1. load and pool --------------------------------------------------------
tr, te = pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)
assert list(tr.columns) == list(te.columns), "train and test CSVs have different columns"
df = pd.concat([tr, te], ignore_index=True); n_train = len(tr); del tr, te
cols = [str(c) for c in df.columns]
first_t = next(i for i, c in enumerate(cols) if c.startswith("leak_C_"))
inp_cols, tgt_cols = cols[:first_t], cols[first_t:]
assert all(c.startswith("leak_C_") for c in tgt_cols), "unexpected non-leak_C_ column after the first leak_C_ column"
assert "time_of_day" in inp_cols, "column 'time_of_day' not found"
X = np.nan_to_num(df.iloc[:, :first_t].to_numpy(dtype=np.float64), nan=0.0)      # NaN -> 0 as in the original pipeline
Y = df.iloc[:, first_t:].to_numpy(dtype=np.float64); del df
n, d_in = X.shape
t_raw = X[:, inp_cols.index("time_of_day")]
uh = np.unique(t_raw); H = len(uh); hs = np.searchsorted(uh, t_raw)                # hour index 0..H-1
print(f"pooled rows={n} (train {n_train} + test {n - n_train}) | inputs={d_in} | targets={Y.shape[1]} | distinct time_of_day values H={H}")

# ---------------------------------------------- 2. leak scenarios: exact ------------------------------------------------
leak_mask = (Y > 0).any(1)
keys, gid = {}, -np.ones(n, dtype=np.int64)
for i in np.flatnonzero(leak_mask): gid[i] = keys.setdefault(Y[i].tobytes(), len(keys))
n_leak_groups = len(keys)
sizes = np.bincount(gid[leak_mask])
hours_unique = (pd.DataFrame({"g": gid[leak_mask], "h": hs[leak_mask]}).groupby("g")["h"].nunique().to_numpy() == sizes).all()
leak_complete = bool((sizes == H).all() and hours_unique)
print(f"leak scenarios (exact, by leak vector): {n_leak_groups} | rows/scenario min/median/max = {sizes.min()}/{int(np.median(sizes))}/{sizes.max()} | "
      f"every leak scenario has exactly {H} rows, one per hour: {leak_complete}")

# ---------------------------------------------- 3. tools for the zero-leak reconstruction ------------------------------
def hour_standardize(Xs, hh):
    """Per hour, per column z-score (computed on the rows given). A scenario-level offset/scale shows up as a z-value that persists across hours."""
    Z = np.zeros_like(Xs)
    for h in np.unique(hh):
        m = hh == h; mu = Xs[m].mean(0); sd = Xs[m].std(0)
        sd = np.where(sd < 1e-10 * (1.0 + np.abs(mu)), np.inf, sd)                 # constant at this hour -> contributes 0
        Z[m] = (Xs[m] - mu) / sd
    return Z

def icc_per_column(Z, g):
    """One-way intraclass correlation per column: fraction of variance that is BETWEEN groups (1 = pure scenario signature)."""
    order = np.argsort(g, kind="stable"); gs, Zs = g[order], Z[order]
    _, starts, cnt = np.unique(gs, return_index=True, return_counts=True)
    G, N = len(cnt), len(g)
    means = np.add.reduceat(Zs, starts, axis=0) / cnt[:, None]
    ssw = np.add.reduceat(Zs ** 2, starts, axis=0) - cnt[:, None] * means ** 2
    msw = np.maximum(ssw, 0).sum(0) / max(N - G, 1)
    msb = (cnt[:, None] * (means - Zs.mean(0)) ** 2).sum(0) / max(G - 1, 1)
    return np.clip((msb - msw) / (msb + (cnt.mean() - 1) * msw + 1e-300), 0, 1)

weights_from_icc = lambda icc: np.where(icc >= ICC_MIN, icc, 0.0)

def sqdist(a, b):
    return np.maximum((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T, 0.0)

def chain_cluster(Z, hh, w, max_pass=10):
    """Partition rows into scenarios with AT MOST ONE row per hour per scenario (Hungarian assignment per hour, then leave-one-hour-out refinement).
    Returns (labels, ambiguity ratio per row)."""
    cols_w = w > 0; Zw = Z[:, cols_w] * np.sqrt(w[cols_w]); N, p = Zw.shape
    rows_h = [np.flatnonzero(hh == h) for h in range(H)]
    n_c = max(len(r) for r in rows_h); h0 = int(np.argmax([len(r) for r in rows_h]))
    lab = -np.ones(N, dtype=np.int64); S = np.zeros((n_c, p)); cnt = np.zeros(n_c)
    lab[rows_h[h0]] = np.arange(len(rows_h[h0])); np.add.at(S, lab[rows_h[h0]], Zw[rows_h[h0]]); np.add.at(cnt, lab[rows_h[h0]], 1)
    for h in sorted((h for h in range(H) if h != h0), key=lambda h: abs(h - h0)):
        r = rows_h[h]
        if len(r) == 0: continue
        cen = S / np.maximum(cnt, 1)[:, None]
        ri, ci = linear_sum_assignment(sqdist(Zw[r], cen))
        lab[r[ri]] = ci; np.add.at(S, ci, Zw[r[ri]]); np.add.at(cnt, ci, 1)
    def leave_hour_out(h):
        r = rows_h[h]; S2, c2 = S.copy(), cnt.copy()
        np.subtract.at(S2, lab[r], Zw[r]); np.subtract.at(c2, lab[r], 1)
        return r, sqdist(Zw[r], S2 / np.maximum(c2, 1)[:, None])
    for _ in range(max_pass):
        changed = False
        for h in range(H):
            if len(rows_h[h]) == 0: continue
            r, cost = leave_hour_out(h); ri, ci = linear_sum_assignment(cost)
            if (lab[r[ri]] != ci).any():
                changed = True; np.subtract.at(S, lab[r], Zw[r]); np.subtract.at(cnt, lab[r], 1)
                lab[r[ri]] = ci; np.add.at(S, ci, Zw[r[ri]]); np.add.at(cnt, ci, 1)
        if not changed: break
    ratio = np.zeros(N)
    for h in range(H):
        if len(rows_h[h]) == 0: continue
        r, cost = leave_hour_out(h); own = cost[np.arange(len(r)), lab[r]]; cost[np.arange(len(r)), lab[r]] = np.inf
        ratio[r] = own / np.maximum(cost.min(1), 1e-300)
    return lab, ratio

def ari(a, b):
    ct = pd.crosstab(a, b).to_numpy().astype(float); c2 = lambda x: x * (x - 1) / 2
    s, sa, sb, tot = c2(ct).sum(), c2(ct.sum(1)).sum(), c2(ct.sum(0)).sum(), c2(ct.sum())
    exp, mx = sa * sb / tot, (sa + sb) / 2
    return 1.0 if mx == exp else float((s - exp) / (mx - exp))

def exact_recovery(true, pred):
    cnt_pred = np.bincount(pred[pred >= 0], minlength=pred.max() + 1) if (pred >= 0).any() else np.zeros(1, int)
    ok = 0
    for t in np.unique(true):
        m = true == t; labs = np.unique(pred[m])
        ok += int(len(labs) == 1 and labs[0] >= 0 and cnt_pred[labs[0]] == m.sum())
    return ok / len(np.unique(true))

def reconstruct(Xs, hh, w_init, max_iter=5):
    """Cluster with initial weights, then self-train the weights on the clusters found, until the partition stops changing."""
    Z = hour_standardize(Xs, hh); w, prev, aris = w_init, None, []
    lab = ratio = None; first = None
    for _ in range(max_iter):
        lab, ratio = chain_cluster(Z, hh, w)
        if first is None: first = lab.copy()
        if prev is not None:
            aris.append(ari(prev, lab))
            if aris[-1] >= 0.9999: break
        prev = lab.copy()
        icc = icc_per_column(Z, lab)
        if (icc >= ICC_MIN).sum() >= 1: w = weights_from_icc(icc)
    return lab, ratio, w, aris, first

# ---------------------------------------------- 4. calibration on leak scenarios (labels hidden) -----------------------
report = dict(n_rows=int(n), n_train_rows=int(n_train), hours=int(H), n_leak_scenarios=int(n_leak_groups), leak_groups_complete=leak_complete)
leak_rows = np.flatnonzero(leak_mask)
Zleak = hour_standardize(X[leak_rows], hs[leak_rows])
icc_leak = icc_per_column(Zleak, gid[leak_rows]); w0 = weights_from_icc(icc_leak); n_sig = int((w0 > 0).sum())
print(f"signature columns (ICC >= {ICC_MIN}) found on leak scenarios: {n_sig} of {d_in}")
cal = []
if n_sig >= MIN_SIGNATURE_COLS and n_leak_groups >= 20:
    ug = rng.permutation(np.unique(gid[leak_rows])); halves = [ug[: len(ug) // 2], ug[len(ug) // 2:]]
    for fit_g, ev_g in (halves, halves[::-1]):
        fit = leak_rows[np.isin(gid[leak_rows], fit_g)]; ev = leak_rows[np.isin(gid[leak_rows], ev_g)]
        w_fit = weights_from_icc(icc_per_column(hour_standardize(X[fit], hs[fit]), gid[fit]))
        if (w_fit > 0).sum() == 0: cal.append(dict(exact_stage1=0.0, exact_final=0.0, ari_final=0.0)); continue
        lab, _, _, _, first = reconstruct(X[ev], hs[ev], w_fit)
        cal.append(dict(exact_stage1=exact_recovery(gid[ev], first), exact_final=exact_recovery(gid[ev], lab), ari_final=ari(gid[ev], lab)))
    for k, c in enumerate(cal): print(f"calibration fold {k}: exact recovery of held-out leak scenarios = {c['exact_final']:.3f} "
                                       f"(before self-training {c['exact_stage1']:.3f}) | ARI = {c['ari_final']:.4f}")
report["calibration"] = cal

# ---------------------------------------------- 5. zero-leak rows ------------------------------------------------------
zero_rows = np.flatnonzero(~leak_mask); n_zero = len(zero_rows)
cph = np.bincount(hs[zero_rows], minlength=H)
equal_hours = bool(n_zero > 0 and (cph == cph[0]).all())
print(f"zero-leak rows: {n_zero} | rows per hour min/max = {cph.min()}/{cph.max()} | implied zero-leak scenarios = {n_zero / H:.2f}")
lab_z, amb_frac, converged, aris_z, n_cols_z = None, np.nan, False, [], 0
if n_zero and n_sig >= MIN_SIGNATURE_COLS:
    lab_z, ratio_z, w_z, aris_z, _ = reconstruct(X[zero_rows], hs[zero_rows], w0)
    amb_frac = float((ratio_z > AMBIG_RATIO).mean()); converged = bool(len(aris_z) > 0 and aris_z[-1] >= 0.9999)
    n_cols_z = int((w_z > 0).sum())
    print(f"zero-leak clustering: {len(np.unique(lab_z))} clusters | self-training ARI between successive partitions = {np.round(aris_z, 5).tolist()} | "
          f"signature columns after self-training = {n_cols_z} | ambiguous rows = {100 * amb_frac:.2f}%")

# ---------------------------------------------- 6. verdict -----------------------------------------------------------------
cal_ok = len(cal) == 2 and min(c["exact_final"] for c in cal) >= MIN_EXACT_RECOVERY
checks = {
    f"every leak scenario is complete (exactly {H} rows, one per hour) -> pooled data are complete": leak_complete,
    "zero-leak rows are spread evenly over the hours (whole scenarios)": equal_hours and n_zero % H == 0,
    f">= {MIN_SIGNATURE_COLS} scenario-signature columns exist (ICC >= {ICC_MIN})": n_sig >= MIN_SIGNATURE_COLS,
    f"calibration: held-out leak scenarios (labels hidden) recovered exactly >= {MIN_EXACT_RECOVERY:.0%}": cal_ok,
    "self-training converged to a stable partition (ARI >= 0.9999)": converged,
    f"ambiguous zero-leak assignments <= {MAX_AMBIGUOUS_FRAC:.1%}": bool(np.isfinite(amb_frac) and amb_frac <= MAX_AMBIGUOUS_FRAC),
}
reliable = all(checks.values())
print("\n================ RELIABILITY OF THE ZERO-LEAK RECONSTRUCTION ================")
for k, v in checks.items(): print(f"  [{'PASS' if v else 'FAIL'}] {k}")
print("  -> " + ("RELIABLE: passes calibration and internal consistency. This is strong evidence, not proof - zero-leak scenarios have no ground truth here."
                 if reliable else "NOT VALIDATED: the zero-leak grouping cannot be trusted."))
policy = ("reconstruct" if reliable else "drop") if ZERO_POLICY == "auto" else ZERO_POLICY
if policy == "reconstruct" and not reliable: warnings.warn("ZERO_POLICY='reconstruct' forced although validation FAILED: zero-leak train/test leakage cannot be excluded.")
print(f"zero-leak policy applied: {policy.upper()}")

# ---------------------------------------------- 7. assemble ids and save ---------------------------------------------------
sid = gid.copy()
if policy == "reconstruct": sid[zero_rows] = n_leak_groups + lab_z
keep = np.ones(n, dtype=bool)
if policy == "drop": keep[zero_rows] = False
X_out, Y_out, sid_out = X[keep], Y[keep], sid[keep]
assert (sid_out >= 0).all()
os.makedirs(OUT_DIR_P2, exist_ok=True)
np.save(os.path.join(OUT_DIR_P2, "pooled_X.npy"), X_out); np.save(os.path.join(OUT_DIR_P2, "pooled_Y.npy"), Y_out)
np.save(os.path.join(OUT_DIR_P2, "pooled_scenario_ids.npy"), sid_out)
with open(os.path.join(OUT_DIR_P2, "pooled_columns.json"), "w") as f:
    json.dump({"input_columns": inp_cols, "target_columns": tgt_cols, "zero_leak_policy": policy}, f)
report.update(checks={k: bool(v) for k, v in checks.items()}, reliable=bool(reliable), policy=policy, n_zero_rows=int(n_zero),
              n_rows_saved=int(keep.sum()), ambiguous_fraction=None if not np.isfinite(amb_frac) else amb_frac, self_training_ari=[float(a) for a in aris_z])
with open(os.path.join(OUT_DIR_P2, "scenario_reconstruction_report.json"), "w") as f: json.dump(report, f, indent=2)
print(f"\nSaved {keep.sum()} rows ({len(np.unique(sid_out))} scenarios) to {OUT_DIR_P2} in {time.time() - t_start:.0f}s "
      f"[{'zero-leak rows DROPPED: ' + str(n_zero) if policy == 'drop' else 'zero-leak scenarios reconstructed'}]. Report: scenario_reconstruction_report.json")
