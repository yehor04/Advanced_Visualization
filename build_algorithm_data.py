"""Build the 'Be the Algorithm' dataset: real per-video features, fitted true
weights, and real captions. Merges an 'algorithm' block into story_data.json.

Eight tunable features, all from the videos table:
  joy, anger, sadness, surprise  (emotion scores 0-1)
  question  (question_count > 0)
  emoji     (emoji_count, normalized)
  short     (how short the video is)
  ai        (created_by_ai)

The "real algorithm" is a linear regression of log10(play_count) on the
standardized features. The simulator scores the user's weighting against it.
"""
import json
import re

import numpy as np
import pandas as pd

FEATURES = ["joy", "anger", "sadness", "surprise", "question", "emoji", "short", "ai"]
N_SAMPLE = 1500

v = pd.read_parquet("data/videos.parquet",
                    columns=["video_id", "topic", "duration", "desc", "created_by_ai",
                             "question_count", "emoji_count",
                             "joy", "anger", "sadness", "surprise"])
eng = pd.read_parquet("data/video_final_engagement.parquet")[["video_id", "play_count"]]
df = v.merge(eng, on="video_id")
df = df[df.desc.fillna("").str.len() > 0]  # need a caption to surface

feat = pd.DataFrame({
    "joy": df.joy.fillna(0),
    "anger": df.anger.fillna(0),
    "sadness": df.sadness.fillna(0),
    "surprise": df.surprise.fillna(0),
    "question": (df.question_count.fillna(0) > 0).astype(float),
    "emoji": (df.emoji_count.fillna(0) / 3).clip(0, 1),
    "short": (1 - df.duration.fillna(60) / 60).clip(0, 1),
    "ai": df.created_by_ai.fillna(0).clip(0, 1),
})
y = np.log10(df.play_count.clip(lower=1).values)

# fit standardized linear model -> "true" weights
means = feat.mean()
stds = feat.std().replace(0, 1)
Xz = ((feat - means) / stds).values
A = np.column_stack([np.ones(len(Xz)), Xz])
beta, *_ = np.linalg.lstsq(A, y, rcond=None)
coef = beta[1:]
pred = A @ beta
maxR = float(np.corrcoef(pred, y)[0, 1])  # best achievable corr (multiple R)

# map fitted coefficients to slider values (-100..100), scaled to the largest
true_weights = {f: int(round(c / np.abs(coef).max() * 100)) for f, c in zip(FEATURES, coef)}

# stratified sample by topic, capped, with clean captions
def clean(s):
    s = re.sub(r"[<>]", "", str(s)).replace("\n", " ").strip()
    return s[:80]

idx = (df.assign(_f=range(len(df)))
         .groupby("topic", group_keys=False)
         .apply(lambda g: g.sample(min(len(g), N_SAMPLE // df.topic.nunique() + 40), random_state=7)))
sample = idx.head(N_SAMPLE)
srows = sample.index
fsamp = feat.loc[srows]

topics = sorted(df.topic.dropna().unique())
videos = []
for vid in srows:
    r = fsamp.loc[vid]
    videos.append({
        "f": [round(float(r[f]), 2) for f in FEATURES],
        "p": int(df.loc[vid, "play_count"]),
        "t": topics.index(df.loc[vid, "topic"]) if df.loc[vid, "topic"] in topics else 0,
        "c": clean(df.loc[vid, "desc"]),
    })

out = {
    "features": FEATURES,
    "labels": ["Joy", "Anger / outrage", "Sadness", "Surprise / shock",
               "Asks a question", "Emoji-heavy", "Short & punchy", "AI-generated"],
    "true_weights": true_weights,
    "means": [round(float(means[f]), 3) for f in FEATURES],
    "maxR": round(maxR, 3),
    "r2": round(maxR ** 2, 3),
    "topics": topics,
    "videos": videos,
}

story = json.load(open("data/story_data.json"))
story["algorithm"] = out
json.dump(story, open("data/story_data.json", "w"), ensure_ascii=True)

print("videos:", len(videos), "| maxR:", round(maxR, 3), "| R^2:", round(maxR ** 2, 3))
print("true weights:", true_weights)
EOF_GUARD = None
