step 50: muon_lr 0.02, momentum 0.95, adam_lr 0.004, wd 0 — loss fell from 5.57 to 2.72 smoothly with no instability, so keep Muon and nudge the embedding Adam lr up slightly.
step 100: muon_lr 0.025, momentum 0.95, adam_lr 0.004, wd 0 — val 2.30 and still falling steadily with max loss only 2.65, so raise Muon lr a little for faster progress.
step 150: muon_lr 0.025, momentum 0.95, adam_lr 0.004, wd 0 — val dropped 2.30 to 2.16 with no spikes, so keep the same settings.
step 200: muon_lr 0.025, momentum 0.95, adam_lr 0.004, wd 0 — val 2.09 and max train loss 2.14 show stable descent, so hold settings before starting a decay.
step 250: muon_lr 0.018, momentum 0.95, adam_lr 0.003, wd 0 — val gains are slowing (2.09 to 2.03), so begin decaying learning rates to settle toward a minimum.
step 300: muon_lr 0.012, momentum 0.95, adam_lr 0.002, wd 0 — val 2.03 to 2.00 with the lower lr still improving, so keep decaying.
step 350: muon_lr 0.005, momentum 0.95, adam_lr 0.001, wd 0 — val fell 2.00 to 1.93 during decay, so anneal further for the final segment.
step 400: training finished, final val 1.9004 — no further settings needed; the annealed lr (0.005 Muon, 0.001 Adam) gave the last drop from 1.93 to 1.90.
