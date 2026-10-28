step 100: muon_lr 0.04, momentum 0.95, adam_lr 0.008, wd 0 — loss falling smoothly with no spikes at default; model is undertrained for 400 steps, so raise both lrs to move faster before a later cooldown
step 200: muon_lr 0.025, momentum 0.95, adam_lr 0.005, wd 0 — higher lr cut val 2.31->2.10 with only a small train spike (2.43); start a stepped cooldown over the last half
step 300: muon_lr 0.008, momentum 0.95, adam_lr 0.0015, wd 0 — val still dropping (2.10->1.98); final segment is a low-lr cooldown to settle into the lowest loss
