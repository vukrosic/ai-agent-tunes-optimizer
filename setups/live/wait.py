"""python3 wait.py --until S   wait (at most 100 s) until training has reached step S (or finished), then print the log rows since the last call."""
import argparse, json, os, time
RUN = os.path.dirname(os.path.abspath(__file__)); log = os.path.join(RUN, "log.jsonl"); seen = os.path.join(RUN, ".seen")
ap = argparse.ArgumentParser(); ap.add_argument("--until", type=int, required=True); a = ap.parse_args()
def rows(): return [json.loads(l) for l in open(log)] if os.path.exists(log) else []
t0 = time.time()
while time.time() - t0 < 100:
    r = rows()
    if r and ("final_val" in r[-1] or (a.until < 400 and r[-1]["step"] >= a.until)): break
    time.sleep(0.5)
r = rows(); k = int(open(seen).read()) if os.path.exists(seen) else 0; open(seen, "w").write(str(len(r)))
for x in r[k:]: print(json.dumps(x))
if not r or "final_val" not in r[-1]: print("(training still running)")
