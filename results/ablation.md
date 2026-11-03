| Variant | Params | vacuum FDE median | drag FDE median | drag_wind FDE median |
|---|---:|---:|---:|---:|
| GRU, per-step displacement output (original design) | 397k | 4.5 px | 6.6 px | 12.6 px |
| Mamba, per-step displacement output | 133k | 8.6 px | 9.0 px | 15.1 px |
| Mamba, displacement output + larger Δ init + context queries | 133k | 17.1 px | 13.5 px | 20.3 px |
| GRU, acceleration output (final) | 397k | 4.7 px | 6.9 px | 12.9 px |
| Mamba, acceleration output (final) | 133k | 6.3 px | 7.5 px | 14.5 px |
