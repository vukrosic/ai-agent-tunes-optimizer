step 50: muon_lr 0.04, momentum 0.95, adam_lr 0.006, wd 0 — loss falling smoothly with no spikes, so push lr up while early in training.
step 100: muon_lr 0.04, momentum 0.95, adam_lr 0.006, wd 0 — higher lr was stable (val 2.66→2.27), keep it for one more segment before cooling down.
step 150: muon_lr 0.036, momentum 0.95, adam_lr 0.0054, wd 0 — progress slowing (val 2.135), start a linear cooldown to ~0 by step 400 (segment midpoint values).
step 200: muon_lr 0.028, momentum 0.95, adam_lr 0.0042, wd 0 — val still dropping steadily (2.063), continue the linear cooldown.
step 250: muon_lr 0.02, momentum 0.95, adam_lr 0.003, wd 0 — val 2.005, gains shrinking; keep decaying lr to settle into the minimum.
step 300: muon_lr 0.012, momentum 0.95, adam_lr 0.0018, wd 0 — val 1.976, cooldown is still paying off; continue linear decay.
step 350: muon_lr 0.004, momentum 0.95, adam_lr 0.0006, wd 0 — final segment, drop lr near zero to finish the cooldown (val fell 0.055 last segment).
