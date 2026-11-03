| Model | Params | vacuum FDE (median / mean) | drag FDE (median / mean) | drag_wind FDE (median / mean) |
|---|---:|---:|---:|---:|
| Constant velocity | 0 | 254.8 / 254.7 px | 152.9 / 154.4 px | 156.3 / 156.3 px |
| Parabola fit (physics) | 0 | 6.4 / 6.9 px | 78.8 / 81.5 px | 79.0 / 82.4 px |
| GRU seq2seq | 397k | **4.7 / 6.9 px** | **6.9 / 9.5 px** | **12.9 / 16.4 px** |
| Mamba-SSM | 133k | 6.3 / 8.8 px | 7.5 / 9.9 px | 14.5 / 18.0 px |
