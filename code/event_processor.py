import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, periodogram
import json

# 1. Grouping

def group_uwb_events_by_time(sensor, min_gap=1.0, min_points=2, win_size_sec=1.0):
    uwbDis = np.array(sensor['uwbDis'])
    timestamps = np.array(sensor['timestampCal'])
    tagId = np.array(sensor['tagId'])
    Fs = sensor['sampleRate']
    X = np.array(sensor['Accel_WR_X_cal'])
    Y = np.array(sensor['Accel_WR_Y_cal'])
    Z = np.array(sensor['Accel_WR_Z_cal'])
    nonzero = uwbDis != 0
    idxs = np.where(nonzero)[0]
    valid_t = timestamps[nonzero]
    valid_d = uwbDis[nonzero]
    valid_tag = tagId[nonzero]
    win_pts = int(Fs * win_size_sec // 2)
    groups = []
    n = len(valid_t)
    if n < min_points:
        return []
    gstart = 0
    for i in range(1, n):
        gap = valid_t[i] - valid_t[i-1]
        tag_mismatch = i<=n-3 and np.all(valid_tag[gstart] != valid_tag[i:i+3])
        if gap >= min_gap or tag_mismatch:
            gend = i-1
            if gend-gstart+1 >= min_points:
                sidx = max(idxs[gstart]-win_pts, 0)
                eidx = min(sidx+2*win_pts, len(X)-1)
                groups.append({
                 'timestamps': valid_t[gstart:gend+1].tolist(),
                 'uwbData': valid_d[gstart:gend+1].tolist(),
                 'tagId': valid_tag[gstart:gend+1].tolist(),
                 'accleDataX': X[sidx:eidx].tolist(),
                 'accleDataY': Y[sidx:eidx].tolist(),
                 'accleDataZ': Z[sidx:eidx].tolist(),
                 'accValue': float(np.mean(np.sqrt(X[sidx:eidx]**2 + Y[sidx:eidx]**2 + Z[sidx:eidx]**2)))
                })
            gstart = i
    if n-gstart >= min_points:
        sidx = max(idxs[gstart]-win_pts, 0)
        eidx = min(sidx+2*win_pts, len(X)-1)
        groups.append({ 'timestamps': valid_t[gstart:].tolist(),
         'uwbData': valid_d[gstart:].tolist(), 'tagId': valid_tag[gstart:].tolist(),
         'accleDataX': X[sidx:eidx].tolist(), 'accleDataY': Y[sidx:eidx].tolist(),
         'accleDataZ': Z[sidx:eidx].tolist(),
         'accValue': float(np.mean(np.sqrt(X[sidx:eidx]**2 + Y[sidx:eidx]**2 + Z[sidx:eidx]**2))) })
    return groups

# 2. Dominant Tag Filtering

def filter_by_dominant_tag(events):
    for e in events:
        tag = pd.Series(e['tagId'])
        dom = tag.mode().iloc[0]
        msk = tag == dom
        for k in ['timestamps', 'uwbData', 'tagId']:
            arr = np.array(e[k])
            e[k] = arr[msk].tolist()
    return events

# 3. Noisy Event Filtering

def filter_noisy_events(events, min_points=3, max_dist=60):
    good = []
    for e in events:
        if len(e['uwbData']) < min_points:
            continue
        init_dist = np.mean(e['uwbData'][:min(3, len(e['uwbData']))])
        if init_dist >= max_dist:
            continue
        good.append(e)
    return good

# 4. Debounce Filter

def filter_consecutive(events, min_gap=1.0):
    if len(events) < 2:
        return events
    keep = [events[0]]
    for curr in events[1:]:
        prev = keep[-1]
        is_same_tag = prev['tagId'][0] == curr['tagId'][0]
        gap = curr['timestamps'][0] - prev['timestamps'][-1]
        if is_same_tag and gap < min_gap:
            continue
        keep.append(curr)
    return keep

# 5. Outlier Removal (MAD)

def find_outliers_mad(x, max_out=3, thresh=3):
    x = np.array(x, dtype=float)
    out_idx = []
    for _ in range(max_out):
        if len(x) < 3:
            break
        m = np.median(x)
        mad = np.median(np.abs(x - m)) or 1e-9
        scores = 0.6745 * np.abs(x - m) / mad
        mx = np.argmax(scores)
        if scores[mx] > thresh and x[mx] >= np.mean(x):
            out_idx.append(mx)
            x = np.delete(x, mx)
        else:
            break
    return out_idx

def remove_outliers_mad(events, max_out=3, thresh=3):
    for e in events:
        idx = find_outliers_mad(e['uwbData'], max_out, thresh)
        for k in ['uwbData', 'timestamps']:
            arr = np.array(e[k])
            mask = np.ones(len(arr), dtype=bool)
            mask[idx] = False
            e[k] = arr[mask].tolist()
    return events

# 6. Feature Extraction (for all grouped events)

def extract_features(events, Fs):
    # For brevity, only a subset of features is shown; extend to full 20 features.
    out = []
    Fc = 0.5
    b, a = butter(2, Fc/(Fs/2), btype='low')
    for e in events:
        try:
            X = np.array(e['accleDataX'])
            Y = np.array(e['accleDataY'])
            Z = np.array(e['accleDataZ'])
            N = len(X)
            GravX = filtfilt(b, a, X)
            GravY = filtfilt(b, a, Y)
            GravZ = filtfilt(b, a, Z)
            BodyX, BodyY, BodyZ = X-GravX, Y-GravY, Z-GravZ
            RawMag = np.sqrt(X**2 + Y**2 + Z**2)
            BodyMag = np.sqrt(BodyX**2 + BodyY**2 + BodyZ**2)
            # Example features:
            feat = {
                'mean_X': float(np.mean(X)),
                'std_Y': float(np.std(Y)),
                'energy_BodyMag': float(np.mean(BodyMag**2)),
                # Add remaining features as in your MATLAB implementation
            }
            out.append(feat)
        except Exception:
            out.append({f'f{i}': np.nan for i in range(1, 21)})
    return pd.DataFrame(out).fillna(0)

# Attach predictions

def attach_predictions(events, preds):
    for e, p in zip(events, preds):
        e['prediction'] = str(p)
    return events

# Event counting (stub; expand to categorize)

def count_touches(left, right):
    return {'left': len(left), 'right': len(right)}
