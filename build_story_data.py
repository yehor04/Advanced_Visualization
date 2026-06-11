"""Build all section datasets for 'The TikTok Lottery' scrollytelling page.

Data: lingbow/tiktok-video-engagement-200k (HuggingFace, CC-BY-NC-4.0).
Expects cached parquets in data/ (videos, video_final_engagement,
creator_daily, trajectories_30d). Outputs data/story_data.json.
"""
import json

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

vids = pd.read_parquet('data/videos.parquet',
                       columns=['video_id', 'author_id', 'topic', 'create_date', 'duration',
                                'created_by_ai', 'anger', 'joy', 'surprise', 'sadness', 'disgust', 'fear'])
eng = pd.read_parquet('data/video_final_engagement.parquet')[['video_id', 'play_count']]
cd = pd.read_parquet('data/creator_daily.parquet')
traj = pd.read_parquet('data/trajectories_30d.parquet')

v = vids.merge(eng, on='video_id')
foll = cd.groupby('author_id')['follower_count'].mean().rename('followers')
v = v.merge(foll, on='author_id', how='left')

EMOTIONS = ['joy', 'anger', 'sadness', 'fear', 'disgust', 'surprise']
v['dominant_emotion'] = v[EMOTIONS].idxmax(axis=1)

EXEMPLARS = {
    'winner': '7433460282111954218',      # 14K-follower cook -> 8.4M views
    'slowburn': '7390549874074324270',    # 1,245 views after week 1 -> 12.9M
    'flop_author': '7813727',             # placeholder, replaced below
}
TWO_FACED_AUTHOR = '6760378027173987334'  # 166K followers: 13 videos <50K, two >11M
BIG_FLOP_AUTHOR = None                    # 2.4M followers, videos under 1K views

out = {}

# ---- Section 1: creators scatter ----------------------------------------
per_creator = (v.dropna(subset=['followers'])
                 .groupby('author_id')
                 .agg(n=('video_id', 'count'),
                      followers=('followers', 'first'),
                      med=('play_count', 'median'),
                      best=('play_count', 'max'),
                      worst=('play_count', 'min')))
per_creator = per_creator[per_creator.n >= 5]
big_flop = v[(v.followers > 1_000_000) & (v.play_count < 1000)]
BIG_FLOP_AUTHOR = big_flop.sort_values('followers').iloc[-1].author_id

out['creators'] = [
    {'f': round(r.followers), 'm': round(r.med), 'b': int(r.best), 'w': int(r.worst), 'n': int(r.n),
     'hl': ('giant' if a == '7813727' else
            'twofaced' if a == TWO_FACED_AUTHOR else
            'bigflop' if a == BIG_FLOP_AUTHOR else
            'winner' if a == v.set_index('video_id').loc[EXEMPLARS['winner']].author_id else None)}
    for a, r in per_creator.iterrows()
]
out['creators'] = [{k: x for k, x in c.items() if x is not None} for c in out['creators']]

# ---- Section 2: lifelines -------------------------------------------------
final = traj[30]
bands = pd.cut(np.log10(final.clip(lower=1)), bins=[-1, 2, 3, 4, 5, 6, 9], labels=False)
sample_ids = []
for b, n_take in zip(range(6), [180, 320, 280, 200, 120, 80]):
    ids = final[bands == b].index
    sample_ids += list(rng.choice(ids, size=min(n_take, len(ids)), replace=False))
for ex in [EXEMPLARS['winner'], EXEMPLARS['slowburn']]:
    if ex in traj.index and ex not in sample_ids:
        sample_ids.append(ex)
# a flop lifeline: median-ish video
flop_id = (final[(final > 500) & (final < 800)]).sample(1, random_state=7).index[0]
if flop_id not in sample_ids:
    sample_ids.append(flop_id)

vmeta = v.set_index('video_id')
out['lifelines'] = []
for vid in sample_ids:
    row = traj.loc[vid]
    vals = [int(row[d]) for d in range(31)]
    item = {'v': vals}
    if vid == EXEMPLARS['winner']:
        item['hl'] = 'winner'
    elif vid == EXEMPLARS['slowburn']:
        item['hl'] = 'slowburn'
    elif vid == flop_id:
        item['hl'] = 'flop'
    if 'hl' in item and vid in vmeta.index:
        m = vmeta.loc[vid]
        item['meta'] = {'topic': m.topic, 'date': str(m.create_date),
                        'followers': None if pd.isna(m.followers) else round(m.followers)}
    out['lifelines'].append(item)

# ---- Section 3: the lottery ----------------------------------------------
p = v.play_count
out['lottery'] = {
    'sample': sorted(int(x) for x in rng.choice(p.values, 8000, replace=False)),
    'stats': {'median': int(p.median()), 'mean': round(float(p.mean())),
              'p90': round(float(p.quantile(.9))), 'p99': round(float(p.quantile(.99))),
              'max': int(p.max()), 'n': int(len(p)),
              'top1pct_share': round(float(p[p >= p.quantile(.99)].sum() / p.sum() * 100), 1),
              'under1k_share': round(float((p < 1000).mean() * 100), 1)},
}
bands_def = [(0, 10_000, '<10K'), (10_000, 100_000, '10K–100K'),
             (100_000, 1_000_000, '100K–1M'), (1_000_000, 1e12, '>1M')]
out['lottery']['bands'] = []
for lo, hi, lab in bands_def:
    s = v[(v.followers >= lo) & (v.followers < hi)].play_count
    out['lottery']['bands'].append({
        'band': lab, 'n': int(len(s)), 'med': int(s.median()),
        'under1k': round(float((s < 1000).mean() * 100), 1),
        'over100k': round(float((s > 100_000).mean() * 100), 2),
        'over1m': round(float((s > 1_000_000).mean() * 100), 2)})

# ---- Section 4: what moves the needle -------------------------------------
def med_table(group_col):
    g = v.groupby(group_col).play_count
    t = pd.DataFrame({'med': g.median(), 'p90': g.quantile(.9), 'n': g.count()})
    return [{'k': str(k), 'med': int(r.med), 'p90': int(r.p90), 'n': int(r.n)}
            for k, r in t.sort_values('med', ascending=False).iterrows()]

v['dur_bucket'] = pd.cut(v.duration, [0, 15, 30, 60, 120, 1e4],
                         labels=['<15s', '15–30s', '30–60s', '1–2min', '>2min'])
out['movers'] = {
    'topic': med_table('topic'),
    'emotion': med_table('dominant_emotion'),
    'duration': med_table('dur_bucket'),
    'ai': med_table(v.created_by_ai.map({1.0: 'AI-generated', 0.0: 'Organic'})),
}

# ---- Section 5: sandbox pool ----------------------------------------------
pool = (v.dropna(subset=['dur_bucket', 'topic'])
          .loc[lambda d: d.dominant_emotion.isin(EMOTIONS)]
          .sample(30000, random_state=42))
topics = sorted(v.topic.dropna().unique())
durs = list(v.dur_bucket.cat.categories)
emos = EMOTIONS
out['sandbox'] = {
    'topics': topics, 'durations': [str(d) for d in durs], 'emotions': emos,
    'pool': [[topics.index(r.topic), durs.index(r.dur_bucket), emos.index(r.dominant_emotion),
              int(r.play_count)] for r in pool.itertuples()],
}

with open('data/story_data.json', 'w') as f:
    json.dump(out, f)
import os
print('story_data.json:', os.path.getsize('data/story_data.json') // 1024, 'KB')
print('creators:', len(out['creators']), '| lifelines:', len(out['lifelines']),
      '| pool:', len(out['sandbox']['pool']))
