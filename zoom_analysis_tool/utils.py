import os
import numpy as np
import pandas as pd
from scipy import stats
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sklearn.metrics
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import Span, DatetimeTicker, DatetimeTickFormatter


def load_diffs(path, fps):
    diffs = pd.read_csv(path)
    diffs['cos_sim'] = 1 - diffs.cos_sim  # Invert cosine similarity
    diffs['second'] = diffs.index.to_numpy() // fps
    
    # Discretize
    ddiffs = diffs.groupby(['meeting_id', 'second'], as_index=False)[['l1', 'l2', 'cos_sim']].max()
    ddiffs['elapsed'] = pd.to_datetime(ddiffs.second, unit='s').dt.strftime('%H:%M:%S')
    ddiffs['elapsed_dt'] = pd.to_datetime(ddiffs.second, unit='s')
    
    # Difference
    ddiffs['l1_diff'] = np.maximum(ddiffs.l1.diff(), 0)
    ddiffs['l2_diff'] = np.maximum(ddiffs.l2.diff(), 0)
    ddiffs['cos_sim_diff'] = np.maximum(ddiffs.cos_sim.diff(), 0)
    
    return ddiffs[['meeting_id', 'elapsed_dt', 'l1_diff', 'l2_diff', 'cos_sim_diff']].dropna(subset=['cos_sim_diff'])


def load_slide_changes(path):
    sldf = None
    if os.path.exists(path):
        sldf = pd.read_csv(path)
        sldf['change_time'] = sldf.change_time.str.split(':').apply(lambda x: ':'.join(['0', *x]) if len(x) == 2 else ':'.join(x))
        sldf['change_time_dt'] = pd.to_datetime('1970-01-01' + ' ' + sldf.change_time, format='%Y-%m-%d %H:%M:%S')
        
    return sldf


def load_intervals(path, meeting_id=None):
    idf = pd.read_csv(path)
    for col in ['start', 'stop']:
        time = idf[col].str.split(':').apply(lambda x: ':'.join(['0', *x]) if len(x) == 2 else ':'.join(x))
        time_dt = pd.to_datetime('1970-01-01' + ' ' + time, format='%Y-%m-%d %H:%M:%S')
        idf[f'{col}_dt'] = time_dt
   
    if meeting_id:
        return idf.query(f'meeting_id == {meeting_id}').reset_index(drop=True)
    
    return idf


def filter_video(ddiffs, idf):
    mask = ddiffs.index < 0
    for st, stp in idf[['start_dt', 'stop_dt']].itertuples(index=False):
        mask = mask | ((ddiffs.elapsed_dt >= st) & (ddiffs.elapsed_dt <= stp))
    ddiffs = ddiffs.loc[mask, :].reset_index(drop=True)
    
    return ddiffs

    
def get_signals(ddiffs, threshold_q=None):
        if threshold_q is None:
            threshold = ddiffs.cos_sim_diff.mean()
            threshold_q = stats.percentileofscore(ddiffs.cos_sim_diff, threshold)
        else:
            threshold = ddiffs.cos_sim_diff.quantile(q=threshold_q)
        signals = ddiffs[['elapsed_dt', 'cos_sim_diff']].copy() 
        signals['signal'] = ddiffs.cos_sim_diff >= threshold
        signals.drop(columns='cos_sim_diff', inplace=True)
        
        return signals, threshold_q
    
    
def get_hmm_signals(ddiffs, model):
    signals = ddiffs[['elapsed_dt', 'cos_sim_diff']].copy()
    ts = signals.cos_sim_diff.to_numpy().reshape(-1,1)
    model.fit(ts)
    state_seq = model.predict(ts)
    state_means = model.means_.ravel()

    # Since predict provides no guarantee on the numbering
    # of the states, we need to make sure we assign '0' to
    # the normal state and a '1' to the outlier state.
    # Most of the time predict will return '0' and '1' 
    # according to this assumption, but this block will 
    # ensure that it happens every time.
    normal_state = np.argmin(state_means)
    signals['signal'] = (~(state_seq == normal_state)).astype(int)
    signals.drop(columns='cos_sim_diff', inplace=True)

    return signals


def get_kf_signals(ddiffs, smoother):
    signals = ddiffs[['elapsed_dt', 'cos_sim_diff']].copy()
    ts = signals.cos_sim_diff.to_numpy().reshape(-1,1)
    smoother.smooth(ts)
    low, up = smoother.get_intervals('sigma_interval', n_sigma=0.1)
    signals['signal'] = (ddiffs.cos_sim_diff >= up.ravel()).astype(int)
    signals.drop(columns='cos_sim_diff', inplace=True)

    return signals
    

def plot_slide_diffs(ddiffs, sldf=None, signals=None, score_type='cos_sim', fig_kwargs=None):
    score_type = score_type + '_diff'
        
    if fig_kwargs is None:
        p = figure(plot_width=1400, plot_height=500, x_axis_type='datetime')
    else:
        p = figure(**fig_kwargs)
    p.line(ddiffs.elapsed_dt, ddiffs[score_type], line_width=2, line_alpha=0.5)
    
    if signals is not None:
        signals = signals.merge(ddiffs, how='inner', on='elapsed_dt')
        p.circle(signals.query('signal').elapsed_dt, 
                 signals.query('signal').cos_sim_diff, 
                 alpha=0.25, 
                 color='red',
                 legend_label='prediction'
                )
        
    if sldf is not None:
        for sc in sldf.change_time_dt:
            c = Span(location=sc, dimension='height', line_color='black', line_width=0.5)
            p.add_layout(c)
        # For legend entry
        p.line(ddiffs.elapsed_dt[0], [0], legend_label='actual', line_color="black", line_width=0.5)
    
    p.title.text = "Inverse cosine similarity - diff"
    p.title.text_font_size = '15pt'
    p.xaxis.ticker = DatetimeTicker(desired_num_ticks=30, num_minor_ticks=10)
    p.xaxis.formatter = DatetimeTickFormatter(minutes='%H:%M:%S', minsec='%H:%M:%S', hourmin='%H:%M:%S', seconds='%H:%M:%S', hours='%H')
    p.xaxis.major_label_orientation = "vertical"
    
    return p


def sanitize_signals(ddiffs, sldf, signals):
    # Create master df
    results = ddiffs[['elapsed_dt', 'cos_sim_diff']] \
                .merge(sldf[['change_time_dt']],
                       how='left',
                       left_on='elapsed_dt',
                       right_on='change_time_dt') \
                .merge(signals, 
                       how='left', 
                       on='elapsed_dt')
    results['plus'] = results.change_time_dt.shift()
    results['minus'] = results.change_time_dt.shift(-1)
    results['signal_to_change_time_dt'] = np.where(results.change_time_dt.notna(),
                                                   results.change_time_dt,
                                                   results[['plus', 'minus']].max(axis=1))
   

    # Get max values for dupes
    maxes = results.groupby(['signal', 'signal_to_change_time_dt'], dropna=False, as_index=False) \
                    .cos_sim_diff \
                    .max() \
                    .rename(columns={'cos_sim_diff': 'max_diff'}) \
                    .dropna(subset=['signal_to_change_time_dt'])
    
    
    # Create flag to drop dupe signals
    results = results.merge(maxes, how='left', on=['signal','signal_to_change_time_dt'])
    results.max_diff = np.where(results.max_diff.isna(), results.cos_sim_diff, results.max_diff)
    results['keep_signal'] = (~results.signal) | (results.max_diff == results.cos_sim_diff)
    
    return results.query('keep_signal').reset_index(drop=True)
    

    
def evaluate(ddiffs, sldf, signals): 
    ## Map signals to true slide change times (for those that are within +/- 1 sec)
    ## Drop duplicate signals
    sanitized_signals = sanitize_signals(ddiffs, sldf, signals)
    sanitized_signals['true_pos'] = sanitized_signals.signal & (sanitized_signals[['change_time_dt', 'plus', 'minus']].notna().sum(axis=1) > 0)
    
    ## Calculate scores    
    # TP + FN
    num_slide_changes = sanitized_signals.change_time_dt.notna().sum()
    
    # TN + FP
    num_non_slide_changes = sanitized_signals.shape[0] - num_slide_changes
    
    # TP + FP
    num_signals = sanitized_signals.signal.sum()
    
    # TP
    tp = sanitized_signals.true_pos.sum()
    
    # FP
    fp = num_signals - tp
    
    # FN
    fn = num_slide_changes - tp
    
    # TN
    tn = num_non_slide_changes - fp
    
    scores = {}
    scores['accuracy'] = (tp + tn) / sanitized_signals.shape[0]
    scores['precision'] = tp / num_signals
    scores['recall'] = tp / num_slide_changes
    if scores['precision'] + scores['recall'] == 0:
        scores['f1'] = 0
    else:
        scores['f1'] = 2 * (scores['precision'] * scores['recall']) / (scores['precision'] + scores['recall'])
    
    return scores



