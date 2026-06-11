"""Build hashtag co-occurrence network JSON for the A3 D3 visualisation.

Data: lingbow/tiktok-video-engagement-200k (HuggingFace, CC-BY-NC-4.0)
Output: data/hashtag_network.json  {nodes, links}
Then inject into the HTML:  python3 inject.py
"""
import json
import re
from collections import Counter
from itertools import combinations

import pandas as pd

TOP_N_TAGS = 200      # keep only the N most-used hashtags
MIN_EDGE_WEIGHT = 15  # drop hashtag pairs that co-occur in fewer videos

BASE = "https://huggingface.co/datasets/lingbow/tiktok-video-engagement-200k/resolve/main"

# Generic reach/platform tags carry no meaning and connect everything to
# everything (fyp alone is on 42k videos), so they are excluded.
GENERIC_RE = re.compile(
    r"^(f+y+p*|fy+|foryou\w*|fypp+\w*|viral\w*|trend\w*|xyzbca\w*|goviral\w*"
    r"|pourtoi\w*|parati\w*|perte\w*)$"
)
GENERIC_TAGS = {
    "tiktok", "capcut", "duet", "stitch", "greenscreen", "creatorsearchinsights",
    "tiktokshop", "onthisday", "viral", "trending", "explore", "explorepage",
    "blowthisup", "dontletthisflop", "repost", "video", "videos", "real",
    "reels", "share", "follow", "followme", "fallowme", "like", "likes",
}


def is_generic(tag: str) -> bool:
    return tag in GENERIC_TAGS or bool(GENERIC_RE.match(tag)) or "fyp" in tag or "foryou" in tag


def main():
    videos = pd.read_parquet(f"{BASE}/videos.parquet", columns=["video_id", "hashtags", "topic"])
    daily = pd.read_parquet(f"{BASE}/engagement_daily.parquet", columns=["video_id", "play_count"])
    final_views = daily.groupby("video_id")["play_count"].max().rename("play_count")

    rows = []
    for vid, tags, topic in videos.itertuples(index=False):
        if tags is None or len(tags) == 0:
            continue
        names = {t["hashtag_name"].lower().strip() for t in tags if t["hashtag_name"]}
        rows.extend((vid, n, topic) for n in names if not is_generic(n))
    ht = pd.DataFrame(rows, columns=["video_id", "tag", "topic"])

    top = ht.tag.value_counts().head(TOP_N_TAGS).index
    ht = ht[ht.tag.isin(top)]

    ht_v = ht.merge(final_views, on="video_id", how="left")
    node_stats = ht_v.groupby("tag").agg(
        count=("video_id", "nunique"),
        median_views=("play_count", "median"),
        total_views=("play_count", "sum"),
        topic=("topic", lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown"),
    )

    pairs = Counter()
    for _, grp in ht.groupby("video_id")["tag"]:
        for a, b in combinations(sorted(set(grp)), 2):
            pairs[(a, b)] += 1

    links = [
        {"source": a, "target": b, "weight": w}
        for (a, b), w in pairs.items()
        if w >= MIN_EDGE_WEIGHT
    ]
    linked = {l["source"] for l in links} | {l["target"] for l in links}
    nodes = [
        {
            "id": tag,
            "count": int(r.count),
            "median_views": float(r.median_views or 0),
            "total_views": int(r.total_views or 0),
            "topic": r.topic,
        }
        for tag, r in node_stats.iterrows()
        if tag in linked
    ]

    with open("data/hashtag_network.json", "w") as f:
        json.dump({"nodes": nodes, "links": links}, f)
    print(f"nodes: {len(nodes)}, links: {len(links)}")


if __name__ == "__main__":
    main()
