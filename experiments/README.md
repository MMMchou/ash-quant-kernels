# Experiments (not part of the package)

| file | question | answer |
|---|---|---|
| `exp_layout.py` | Is the AWQ row-major layout the bandwidth limiter? | No: load-only on raw AWQ layout reaches 3.4–4.1 TB/s; Marlin-style tiled repack only +6–15%. |
| `nibble_sliced_v4.py` | Avoid `tl.interleave` by doing 8 nibble-slice dots on the packed tile? | Correct (err 2.7e-4) but slower than v3: 31.7 vs 19.8 us (qkv, M=1). 8 narrow dots + strided scale loads cost more than the layout conversion they remove. |
