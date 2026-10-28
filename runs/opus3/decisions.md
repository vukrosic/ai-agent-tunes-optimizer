step 50: muon_lr 0.04, momentum 0.95, adam_lr 0.006, wd 0 — loss falling smoothly with no spikes, so raise both lrs to train faster in the mid phase.
step 100: muon_lr 0.05, momentum 0.95, adam_lr 0.008, wd 0 — higher lr gave a clean drop (2.66→2.27) with only a small early bump, so push lr a bit higher.
step 150: muon_lr 0.05, momentum 0.95, adam_lr 0.008, wd 0 — still improving (val 2.14) but max loss 2.37 hints at noise, so hold lr rather than raise it.
step 200: muon_lr 0.045, momentum 0.95, adam_lr 0.007, wd 0 — stable progress (val 2.09, max loss near start), keep lr high with a slight trim before the final decay.
step 250: muon_lr 0.03, momentum 0.95, adam_lr 0.005, wd 0 — train loss flattened around 1.92 at high lr, so start the decay over the last 150 steps.
step 300: muon_lr 0.017, momentum 0.95, adam_lr 0.003, wd 0 — decay is paying off (val 2.02→1.98), continue stepping lr down.
step 350: muon_lr 0.006, momentum 0.95, adam_lr 0.001, wd 0 — lower lr gave the largest val drop yet (1.98→1.92), so anneal to near zero for the final segment.
