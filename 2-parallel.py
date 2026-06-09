import os, sys, json, random, subprocess, tempfile, time, traceback
import numpy as np
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ── Config ────────────────────────────────────────────────────────────────────
NET_FILE    = "map.net.xml"
TAZ_FILE    = "grids.taz.xml"
SUMOCFG     = "sim.sumocfg"
POP_SIZE    = 6
GENERATIONS = 30
MUTATE_RATE = 0.05
EARLY_STOP  = 1e-4

def log(msg): print(msg, flush=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def save_od_to_xml(od_matrix, zones, filename):
    root_el  = ET.Element("data")
    interval = ET.SubElement(root_el, "interval", {"begin": "0", "end": "3600"})
    rows, cols = np.nonzero(od_matrix)
    for i, j in zip(rows, cols):
        if i == j: continue
        ET.SubElement(interval, "tazRelation", {
            "from": zones[i], "to": zones[j],
            "count": str(int(od_matrix[i, j])),
        })
    ET.ElementTree(root_el).write(filename, encoding="utf-8", xml_declaration=True)

def write_meandata_config(output_xml, interval_end="3600"):
    """Write a SUMO meandata (edgedata) config that captures speedRelative."""
    root = ET.Element("additional")
    ET.SubElement(root, "edgeData", {
        "id":     "edge_stats",
        "freq":   interval_end,
        "file":   output_xml,
        "excludeEmpty": "true",
    })
    return root

def load_edge_relative_speed(file_path):
    tree = ET.parse(file_path)
    out = {}
    for interval in tree.getroot().findall("interval"):
        for edge in interval.findall("edge"):
            eid = edge.get("id")
            tt  = edge.get("speedRelative") or edge.get("speed")
            if eid and tt:
                out[eid] = float(tt)
    return out

def fitness(sim, target):
    common = set(sim.keys()) & set(target.keys())
    if not common: return 1e9
    diff = np.array([sim[e] - target[e] for e in common])
    return float(np.dot(diff, diff) / len(common))

def shell(cmd, label, step):
    out = [f"[{label}] >>> {step}", f"[{label}]     cmd: {cmd}"]
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        elapsed = time.perf_counter() - t0
        out.append(f"[{label}]     rc={r.returncode}  elapsed={elapsed:.2f}s")
        if r.stdout.strip(): out.append(f"[{label}]     stdout: {r.stdout.strip()[:300]}")
        if r.stderr.strip(): out.append(f"[{label}]     stderr: {r.stderr.strip()[:300]}")
    except subprocess.TimeoutExpired:
        out.append(f"[{label}]     TIMEOUT after 300s")
    except Exception as e:
        out.append(f"[{label}]     EXCEPTION: {e}")
    return "\n".join(out)

# ── Worker — uses abs_* paths passed in so CWD doesn't matter ─────────────────
def evaluate_individual(args):
    idx, od, zones, target, abs_net, abs_taz, abs_sumocfg = args
    label = f"ind{idx}"
    out   = [f"[{label}] ── START  pid={os.getpid()} ──────────"]
    t0    = time.perf_counter()

    try:
        with tempfile.TemporaryDirectory() as tmp:
            out.append(f"[{label}] tmpdir: {tmp}")

            od_xml        = os.path.join(tmp, "od.xml")
            trips_file    = os.path.join(tmp, "trips.xml")
            routes_file   = os.path.join(tmp, "routes.rou.xml")
            edges_file    = os.path.join(tmp, "edges.xml")
            meandata_file = os.path.join(tmp, "meandata.add.xml")

            # Write OD
            save_od_to_xml(od, zones, od_xml)
            out.append(f"[{label}] od.xml written  size={os.path.getsize(od_xml)}")

            # od2trips: -z=OD xml, -n=TAZ file, -o=output trips  (matches original)
            out.append(shell(
                f'od2trips -z "{od_xml}" -n "{abs_taz}" -o "{trips_file}"',
                label, "od2trips"))
            trips_ok = os.path.exists(trips_file)
            out.append(f"[{label}] trips.xml: exists={trips_ok}  size={os.path.getsize(trips_file) if trips_ok else 'MISSING'}")
            if not trips_ok:
                out.append(f"[{label}] ABORT: od2trips failed")
                print("\n".join(out), flush=True)
                return idx, 1e9

            # duarouter
            out.append(shell(
                f'duarouter -n "{abs_net}" -r "{trips_file}" -o "{routes_file}" --ignore-errors true',
                label, "duarouter"))
            routes_ok = os.path.exists(routes_file)
            out.append(f"[{label}] routes.xml: exists={routes_ok}  size={os.path.getsize(routes_file) if routes_ok else 'MISSING'}")
            if not routes_ok:
                out.append(f"[{label}] ABORT: duarouter failed")
                print("\n".join(out), flush=True)
                return idx, 1e9

            # Write meandata additional file so SUMO outputs speedRelative
            meandata_add = os.path.join(tmp, "meandata.add.xml")
            md_root = ET.Element("additional")
            ET.SubElement(md_root, "edgeData", {
                "id": "edge_stats", "freq": "3600",
                "file": edges_file, "excludeEmpty": "true",
            })
            ET.ElementTree(md_root).write(meandata_add, encoding="utf-8", xml_declaration=True)
            out.append(f"[{label}] meandata.add.xml written")

            # SUMO — override routes and add meandata output
            # --route-files overrides whatever is in sumocfg
            out.append(shell(
                f'sumo -c "{abs_sumocfg}"'
                f' --route-files "{routes_file}"'
                f' --additional-files "{meandata_add}"'
                f' --no-step-log true --no-warnings true',
                label, "sumo"))

            edges_ok = os.path.exists(edges_file)
            out.append(f"[{label}] edges.xml: exists={edges_ok}  size={os.path.getsize(edges_file) if edges_ok else 'MISSING'}")
            if not edges_ok:
                out.append(f"[{label}] ABORT: edges file missing")
                print("\n".join(out), flush=True)
                return idx, 1e9

            # Peek at first few lines so we can verify format
            with open(edges_file) as f:
                head = "".join(f.readlines()[:6])
            out.append(f"[{label}] edges.xml head:\n{head}")

            sim   = load_edge_relative_speed(edges_file)
            score = fitness(sim, target)
            out.append(f"[{label}] sim edges={len(sim)}  common={len(set(sim)&set(target))}  score={score:.6f}")

    except Exception as e:
        out.append(f"[{label}] UNHANDLED EXCEPTION: {e}\n{traceback.format_exc()}")
        score = 1e9

    out.append(f"[{label}] ── DONE  total={time.perf_counter()-t0:.2f}s ──────────")
    print("\n".join(out), flush=True)
    return idx, score

# ── GA operators ──────────────────────────────────────────────────────────────
def random_od(n):
    m = np.random.randint(0, 101, size=(n, n))
    np.fill_diagonal(m, 0)
    return m

def crossover(a, b):
    mask = np.random.rand(*a.shape) < 0.5
    child = np.where(mask, a, b)
    np.fill_diagonal(child, 0)
    return child

def mutate(m, rate=MUTATE_RATE):
    m = m.copy()
    mask = np.random.rand(*m.shape) < rate
    np.fill_diagonal(mask, False)
    m[mask] = np.random.randint(0, 101, size=mask.sum())
    return m

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Resolve absolute paths once here — workers inherit these as plain strings
    CWD        = os.path.abspath(".")
    abs_net    = os.path.abspath(NET_FILE)
    abs_taz    = os.path.abspath(TAZ_FILE)
    abs_sumocfg= os.path.abspath(SUMOCFG)

    log(f"CWD:     {CWD}")
    log(f"NET:     {abs_net}  exists={os.path.exists(abs_net)}")
    log(f"TAZ:     {abs_taz}  exists={os.path.exists(abs_taz)}")
    log(f"SUMOCFG: {abs_sumocfg}  exists={os.path.exists(abs_sumocfg)}")

    log("Loading TAZ zones...")
    tree  = ET.parse(abs_taz)
    zones = [t.attrib["id"] for t in tree.getroot().findall("taz")]
    n     = len(zones)
    log(f"  zones: {n}  first 3: {zones[:3]}")

    log("Loading neshan weights...")
    neshan_weights = json.load(open("data/neshan-sample-weights.json"))
    log(f"  target edges: {len(neshan_weights)}  sample: {list(neshan_weights.items())[:3]}")

    WORKERS = min(POP_SIZE, multiprocessing.cpu_count())
    log(f"  workers: {WORKERS}")

    log("\n── Sanity check: running ind0 directly (no pool) ──")
    _, score0 = evaluate_individual(
        (0, random_od(n), zones, neshan_weights, abs_net, abs_taz, abs_sumocfg))
    log(f"── Sanity check done  score={score0} ──\n")

    population    = [random_od(n) for _ in range(POP_SIZE)]
    best_solution = None
    best_score    = float("inf")

    t_total   = time.perf_counter()
    gen_times = []  # (gen, elapsed, gen_best, all_time_best)

    for gen in range(GENERATIONS):
        t_gen = time.perf_counter()
        log(f"\n=== Generation {gen} ===  pop={POP_SIZE}  workers={WORKERS}")

        args   = [(i, ind, zones, neshan_weights, abs_net, abs_taz, abs_sumocfg)
                  for i, ind in enumerate(population)]
        scores = [1e9] * POP_SIZE

        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(evaluate_individual, a): a[0] for a in args}
            log(f"  {POP_SIZE} futures submitted, waiting...")
            for fut in as_completed(futures):
                try:
                    idx, score = fut.result()
                    scores[idx] = score
                    log(f"  → ind{idx} score={score:.6f}  ({sum(s<1e9 for s in scores)}/{POP_SIZE} done)")
                except Exception as e:
                    idx = futures[fut]
                    log(f"  → ind{idx} EXCEPTION: {e}\n{traceback.format_exc()}")

        log(f"  Gen {gen} done in {time.perf_counter()-t_gen:.1f}s  scores={[f'{s:.4f}' for s in scores]}")

        for ind, score in zip(population, scores):
            if score < best_score:
                best_score    = score
                best_solution = ind.copy()
        log(f"  All-time best: {best_score:.6f}")
        gen_times.append((gen, time.perf_counter()-t_gen, min(scores), best_score))

        if best_score <= EARLY_STOP:
            log(f"  Early stop — score {best_score} <= {EARLY_STOP}")
            break

        order    = np.argsort(scores)
        selected = [population[i] for i in order[:POP_SIZE // 2]]
        new_pop  = []
        while len(new_pop) < POP_SIZE:
            p1, p2 = random.sample(selected, 2)
            new_pop.append(mutate(crossover(p1, p2)))
        population = new_pop

    total_elapsed = time.perf_counter() - t_total
    log(f"\n{'='*60}")
    log(f"  TIMING SUMMARY")
    log(f"{'='*60}")
    log(f"  {'Gen':>4}  {'Time(s)':>8}  {'Gen Best':>12}  {'All-time Best':>14}")
    log(f"  {'-'*4}  {'-'*8}  {'-'*12}  {'-'*14}")
    for g, t, gb, ab in gen_times:
        log(f"  {g:>4}  {t:>8.1f}  {gb:>12.6f}  {ab:>14.6f}")
    log(f"{'='*60}")
    log(f"  Total wall time : {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)")
    log(f"  Generations run : {len(gen_times)}")
    log(f"  Avg gen time    : {total_elapsed/max(len(gen_times),1):.1f}s")
    log(f"  Best score      : {best_score:.6f}")
    log(f"{'='*60}")
    log(f"\nDONE — best score: {best_score:.6f}")
    np.save("best_od.npy", best_solution)

    root_el  = ET.Element("data")
    interval = ET.SubElement(root_el, "interval", {"begin": "0", "end": "3600"})
    rows, cols = np.nonzero(best_solution)
    for i, j in zip(rows, cols):
        if i == j: continue
        ET.SubElement(interval, "tazRelation", {
            "from": zones[i], "to": zones[j],
            "count": str(int(best_solution[i, j])),
        })
    ET.ElementTree(root_el).write("final_od.xml", encoding="utf-8", xml_declaration=True)
    log("Saved final_od.xml")
    
    
    
