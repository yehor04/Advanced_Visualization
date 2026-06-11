# The TikTok Lottery — why followers don't buy you views

**Live page: https://yehor04.github.io/Advanced_Visualization/**

A scrollytelling data story built for Assignment 3 of *Advanced Interactive Visualization*
(JKU Linz, June 2026) by **Group 4**: Yehor Larcenko, Olha, Ivanna, Davyd.

We follow 209,543 TikTok videos from 1,872 creators — each tracked day by day for 30 days
after posting — and ask one question: does follower count actually buy you views?
Short answer: it buys you lottery tickets, not a prize. The median video ends its month
at 641 views while the top 1% of videos collects 64% of all views in the dataset.

## How it's built

- **D3.js v7** for all graphics (animated log-log scatter, 30-day "lifelines",
  unit/waffle charts, an interactive "post 100 videos" simulator)
- **Scrollama** for the scroll-driven narrative steps
- No frameworks, no build step — one self-contained `index.html` with the data embedded

## Repository layout

| File | Purpose |
|---|---|
| `index.html` | the deployed page (template + embedded data) |
| `story_template.html` | page source with a `__DATA__` placeholder |
| `build_story_data.py` | data pipeline: HuggingFace tables → `data/story_data.json` |
| `data/story_data.json` | the five section datasets used by the page |
| `archive/` | rejected first design iteration (hashtag co-occurrence network) |

## Rebuilding

```bash
pip install pandas pyarrow
python3 build_story_data.py     # writes data/story_data.json (downloads ~240MB once)
python3 - << 'EOF'
data = open('data/story_data.json').read()
html = open('story_template.html').read()
open('index.html', 'w').write(html.replace('__DATA__', data))
EOF
```

## Data

[lingbow/tiktok-video-engagement-200k](https://huggingface.co/datasets/lingbow/tiktok-video-engagement-200k)
(CC BY-NC 4.0): videos posted 24 June – 9 November 2024, with daily engagement and
follower panels. Engagement numbers are platform data; topic and emotion labels are
model-derived by the dataset authors. Views = cumulative play count on the last observed
day; lifelines use the 73,940 videos tracked for ≥30 days; medians are used throughout
because of the extreme skew.
