step 50: muon_lr 0.035, momentum 0.95, adam_lr 0.005, wd 0 — loss fell smoothly 5.57→2.72 with no spikes, so a tiny model can likely take a higher lr during the fast phase.
step 100: muon_lr 0.04, momentum 0.95, adam_lr 0.006, wd 0 — higher lr was stable (max 2.68, val 2.29), so push slightly higher before starting the decay.
step 150: muon_lr 0.04, momentum 0.95, adam_lr 0.006, wd 0 — still stable and improving (val 2.14), hold the peak lr one more segment before decaying.
step 200: muon_lr 0.03, momentum 0.95, adam_lr 0.0045, wd 0 — halfway point, start a stepwise decay toward ~0 by step 400 to cash in noise reduction.
step 250: muon_lr 0.022, momentum 0.95, adam_lr 0.0033, wd 0 — val still dropping 0.07 per segment, continue the gradual decay.
step 300: muon_lr 0.014, momentum 0.95, adam_lr 0.0021, wd 0 — gains slowing (val -0.037), keep decaying so the last segment can anneal.
step 350: muon_lr 0.006, momentum 0.95, adam_lr 0.0009, wd 0 — last segment, anneal to a low lr since decay gave the biggest val drop so far (-0.045).
step 400: training finished — final val 1.8884.
