import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, periodogram

def group_uwb_events_by_time(sensor, Fs, min_gap=1.0, min_points=2, win_size_sec=1.0):
    uwbDis = np.array(sensor['uwbDis'])
    timestamps = np.array(sensor['timestampCal'])
    tagId = np.array(sensor['tagId'])
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
        tag_mismatch = i <= n-3 and np.all(valid_tag[gstart] != valid_tag[i:i+3])
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
        groups.append({
            'timestamps': valid_t[gstart:].tolist(),
            'uwbData': valid_d[gstart:].tolist(),
            'tagId': valid_tag[gstart:].tolist(),
            'accleDataX': X[sidx:eidx].tolist(),
            'accleDataY': Y[sidx:eidx].tolist(),
            'accleDataZ': Z[sidx:eidx].tolist(),
            'accValue': float(np.mean(np.sqrt(X[sidx:eidx]**2 + Y[sidx:eidx]**2 + Z[sidx:eidx]**2)))
        })
    return groups

def filter_by_dominant_tag(events):
    for e in events:
        tag = pd.Series(e['tagId'])
        dom = tag.mode().iloc[0]
        msk = tag == dom
        for k in ['timestamps', 'uwbData', 'tagId']:
            arr = np.array(e[k])
            e[k] = arr[msk].tolist()
    return events

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

def find_outliers_mad(x, max_out=3, thresh=3):
    x = np.array(x, dtype=float)
    out_idx = []
    orig = x.copy()
    for _ in range(max_out):
        if len(x) < 3:
            break
        m = np.median(x)
        mad = np.median(np.abs(x - m)) or 1e-9
        scores = 0.6745 * np.abs(x - m) / mad
        mx = np.argmax(scores)
        if scores[mx] > thresh and x[mx] >= np.mean(orig):
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

def extract_features(events, Fs):
    # feat_names = [
    #     'tBody_IQR_Y', 'tRaw_IQR_Mag', 'fBody_Energy_Y', 'tGrav_Mean_X',
    #     'tBody_SMA', 'tRaw_Mean_X', 'tBody_Mean_Mag', 'tRaw_Std_Y',
    #     'tRaw_Corr_XZ', 'tGrav_Range_Mag', 'tBody_Corr_XZ', 'tGrav_Range_Y',
    #     'tGrav_Std_Y', 'tRaw_IQR_Y', 'tBody_Range_Y', 'tRaw_Range_Y',
    #     'tBody_IQR_Z', 'tBody_Std_Y', 'tBody_Energy_Mag', 'fBody_Energy_Mag'
    # ]
    feat_names = [
    'tRaw_IQR_Mag', 'tBody_IQR_Y', 'fBody_Energy_Y', 'tBody_SMA', 'tGrav_Mean_X',
    'tRaw_Mean_X', 'tRaw_Corr_XZ', 'tRaw_Std_Y', 'tBody_Mean_Mag', 'tRaw_IQR_Y',
    'tBody_Corr_XZ', 'tGrav_Range_Y', 'tGrav_Range_Mag', 'tGrav_Std_Y', 'tRaw_Range_Y',
    'tBody_IQR_Z', 'tBody_Range_Y', 'tBody_Std_Y', 'fBody_Energy_Mag', 'tBody_Energy_Mag'
    ]
    out = []
    Fc = 0.5
    order = 2
    b, a = butter(order, Fc/(Fs/2), btype='low')
    for e in events:
        try:
            X = np.array(e['accleDataX'])
            Y = np.array(e['accleDataY'])
            Z = np.array(e['accleDataZ'])
            N = len(X)
            if N < order * 3 + 1:
                out.append({k: np.nan for k in feat_names})
                continue
            GravX = filtfilt(b, a, X)
            GravY = filtfilt(b, a, Y)
            GravZ = filtfilt(b, a, Z)
            BodyX, BodyY, BodyZ = X-GravX, Y-GravY, Z-GravZ
            RawMag = np.sqrt(X**2 + Y**2 + Z**2)
            GravMag = np.sqrt(GravX**2 + GravY**2 + GravZ**2)
            BodyMag = np.sqrt(BodyX**2 + BodyY**2 + BodyZ**2)
            v = {}
            v['tBody_IQR_Y'] = float(np.subtract(*np.percentile(BodyY, [75,25])))
            v['tRaw_IQR_Mag'] = float(np.subtract(*np.percentile(RawMag, [75,25])))
            fy, _ = periodogram(BodyY, [], N, Fs, 'power')
            v['fBody_Energy_Y'] = float(np.sum(fy))
            v['tGrav_Mean_X'] = float(np.mean(GravX))
            v['tBody_SMA'] = float(np.mean(np.abs(BodyX) + np.abs(BodyY) + np.abs(BodyZ)))
            v['tRaw_Mean_X'] = float(np.mean(X))
            v['tBody_Mean_Mag'] = float(np.mean(BodyMag))
            v['tRaw_Std_Y'] = float(np.std(Y))
            v['tRaw_Corr_XZ'] = float(np.corrcoef(X, Z)[0,1])
            v['tGrav_Range_Mag'] = float(np.max(GravMag) - np.min(GravMag))
            v['tBody_Corr_XZ'] = float(np.corrcoef(BodyX, BodyZ)[0,1])
            v['tGrav_Range_Y'] = float(np.max(GravY) - np.min(GravY))
            v['tGrav_Std_Y'] = float(np.std(GravY))
            v['tRaw_IQR_Y'] = float(np.subtract(*np.percentile(Y, [75,25])))
            v['tBody_Range_Y'] = float(np.max(BodyY) - np.min(BodyY))
            v['tRaw_Range_Y'] = float(np.max(Y) - np.min(Y))
            v['tBody_IQR_Z'] = float(np.subtract(*np.percentile(BodyZ, [75,25])))
            v['tBody_Std_Y'] = float(np.std(BodyY))
            v['tBody_Energy_Mag'] = float(np.mean(BodyMag**2))
            fmag, _ = periodogram(BodyMag, [], N, Fs, 'power')
            v['fBody_Energy_Mag'] = float(np.sum(fmag))
            out.append(v)
        except Exception:
            out.append({k: np.nan for k in feat_names})
    return pd.DataFrame(out).fillna(0)

def attach_predictions(events, preds):
    for e, p in zip(events, preds):
        e['prediction'] = str(p)
    return events

def count_touches_by_hand(left, right, time_thresh=2.0, range_thresh=20):
    counts = {'left': 0, 'right': 0, 'simultaneous_both': 0, 'simultaneous_none': 0}
    left_idx = right_idx = 0
    event_log = []
    while left_idx < len(left) and right_idx < len(right):
        if not left[left_idx]['timestamps']:
            left_idx += 1
            continue
        if not right[right_idx]['timestamps']:
            right_idx += 1
            continue
        ltime = left[left_idx]['timestamps'][0]
        rtime = right[right_idx]['timestamps'][0]
        lid = left[left_idx]['tagId'][0]
        rid = right[right_idx]['tagId'][0]
        lrange = np.mean(left[left_idx]['uwbData'][:min(3, len(left[left_idx]['uwbData']))])
        rrange = np.mean(right[right_idx]['uwbData'][:min(3, len(right[right_idx]['uwbData']))])
        lp = left[left_idx].get('prediction', None)
        rp = right[right_idx].get('prediction', None)
        dt = rtime - ltime
        if abs(dt) <= time_thresh:
            start_time = min(ltime, rtime)
            if lid == rid:
                if abs(lrange - rrange) <= range_thresh and max(lrange, rrange) < range_thresh:
                    tp = 'simultaneous_both'
                    tag2 = lid
                elif lrange < rrange:
                    tp = 'simultaneous_left'
                    tag2 = lid
                else:
                    tp = 'simultaneous_right'
                    tag2 = rid
                if lp == '0' and rp == '1':
                    tp = 'right_only'; tag2 = rid
                elif lp == '1' and rp == '0':
                    tp = 'left_only'; tag2 = lid
                elif lp == '0' and rp == '0':
                    tp = 'simultaneous_none'; tag2 = lid
                event_log.append({'type': tp, 'tagId': tag2, 'startTime': start_time,
                                  'leftPrediction': lp, 'rightPrediction': rp,
                                  'leftIndex': left_idx, 'leftValue': lrange, 'leftTagId': lid,
                                  'rightIndex': right_idx, 'rightValue': rrange, 'rightTagId': rid})
                left_idx += 1
                right_idx += 1
            elif dt < 0:
                event_log.append({'type': 'left_only', 'tagId': lid, 'startTime': ltime,
                                  'leftPrediction': lp, 'rightPrediction': None,
                                  'leftIndex': left_idx, 'leftValue': lrange, 'leftTagId': lid,
                                  'rightIndex': None, 'rightValue': None, 'rightTagId': None})
                left_idx += 1
            else:
                event_log.append({'type': 'right_only', 'tagId': rid, 'startTime': rtime,
                                  'leftPrediction': None, 'rightPrediction': rp,
                                  'leftIndex': None, 'leftValue': None, 'leftTagId': None,
                                  'rightIndex': right_idx, 'rightValue': rrange, 'rightTagId': rid})
                right_idx += 1
        elif dt < 0:
            event_log.append({'type': 'right_only', 'tagId': rid, 'startTime': rtime,
                              'leftPrediction': None, 'rightPrediction': rp,
                              'leftIndex': None, 'leftValue': None, 'leftTagId': None,
                              'rightIndex': right_idx, 'rightValue': rrange, 'rightTagId': rid})
            right_idx += 1
        else:
            event_log.append({'type': 'left_only', 'tagId': lid, 'startTime': ltime,
                              'leftPrediction': lp, 'rightPrediction': None,
                              'leftIndex': left_idx, 'leftValue': lrange, 'leftTagId': lid,
                              'rightIndex': None, 'rightValue': None, 'rightTagId': None})
            left_idx += 1
    while left_idx < len(left):
        if not left[left_idx]['timestamps']:
            left_idx += 1
            continue
        ltime = left[left_idx]['timestamps'][0]
        lid = left[left_idx]['tagId'][0]
        lrange = np.mean(left[left_idx]['uwbData'][:min(3, len(left[left_idx]['uwbData']))])
        lp = left[left_idx].get('prediction', None)
        event_log.append({'type': 'left_only', 'tagId': lid, 'startTime': ltime,
                          'leftPrediction': lp, 'rightPrediction': None,
                          'leftIndex': left_idx, 'leftValue': lrange, 'leftTagId': lid,
                          'rightIndex': None, 'rightValue': None, 'rightTagId': None})
        left_idx += 1
    while right_idx < len(right):
        if not right[right_idx]['timestamps']:
            right_idx += 1
            continue
        rtime = right[right_idx]['timestamps'][0]
        rid = right[right_idx]['tagId'][0]
        rrange = np.mean(right[right_idx]['uwbData'][:min(3, len(right[right_idx]['uwbData']))])
        rp = right[right_idx].get('prediction', None)
        event_log.append({'type': 'right_only', 'tagId': rid, 'startTime': rtime,
                          'leftPrediction': None, 'rightPrediction': rp,
                          'leftIndex': None, 'leftValue': None, 'leftTagId': None,
                          'rightIndex': right_idx, 'rightValue': rrange, 'rightTagId': rid})
        right_idx += 1
    types = [e['type'] for e in event_log]
    counts['left'] = sum(t in ['left_only', 'simultaneous_left'] for t in types)
    counts['right'] = sum(t in ['right_only', 'simultaneous_right'] for t in types)
    counts['simultaneous_both'] = sum(t == 'simultaneous_both' for t in types)
    counts['simultaneous_none'] = sum(t == 'simultaneous_none' for t in types)
    return counts, event_log

def event_log_finalized(event_log):
    finalized_log = list(event_log)
    default_counts = {'left': 0, 'right': 0, 'simultaneous_both': 0, 
                      'simultaneous_none': 0, 'consecutive_touch': 0}
    if len(finalized_log) < 2:
        return finalized_log, default_counts
    time_threshold = 20.0
    for i in range(len(finalized_log)-1):
        a = finalized_log[i]
        b = finalized_log[i+1]
        dt = b['startTime'] - a['startTime']
        if dt > time_threshold:
            continue
        if a['tagId'] != b['tagId']:
            continue
        is_left_b = b['type'] in ['left_only', 'simultaneous_left', 'simultaneous_both', 'simultaneous_none']
        is_right_b = b['type'] in ['right_only', 'simultaneous_right', 'simultaneous_both', 'simultaneous_none']
        is_left_a = a['type'] in ['left_only', 'simultaneous_left', 'simultaneous_both', 'simultaneous_none']
        is_right_a = a['type'] in ['right_only', 'simultaneous_right', 'simultaneous_both', 'simultaneous_none']
        is_consecutive_a = a['type'] == 'consecutive_touch'
        if (is_left_a or (is_consecutive_a and is_left_b)) and is_left_b and b.get('leftPrediction', None) == '0':
            finalized_log[i+1]['type'] = 'consecutive_touch'
        elif (is_right_a or (is_consecutive_a and is_right_b)) and is_right_b and b.get('rightPrediction', None) == '0':
            finalized_log[i+1]['type'] = 'consecutive_touch'
    types = [e['type'] for e in finalized_log]
    default_counts['left'] = sum(t in ['left_only', 'simultaneous_left'] for t in types)
    default_counts['right'] = sum(t in ['right_only', 'simultaneous_right'] for t in types)
    default_counts['simultaneous_both'] = sum(t == 'simultaneous_both' for t in types)
    default_counts['consecutive_touch'] = sum(t == 'consecutive_touch' for t in types)
    default_counts['simultaneous_none'] = sum(t == 'simultaneous_none' for t in types)
    return finalized_log, default_counts
