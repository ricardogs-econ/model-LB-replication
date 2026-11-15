"""The runner of the section scripts (replicate_sec_*.py, replicate_sup_S*.py and
replicate_all.py): it runs the steps of mlb/steps.py and compares what they write with
reference/.

Quick mode (the default; also named --quick). Each step runs one module of mlb/. A
step whose source is 'recomputed' computes its result again (an exact computation, or a
small simulation with the recorded seeds) and compares it with reference/; a step whose
source is 'reference' shows a result from reference/ (it draws a figure or renders a
table from it, or rebuilds a table from other reference files, and checks what it
draws) and --full regenerates that result. Every step prints its command, its wall time
and its result; its console goes to output/replicate/logs/<step>.log, its files to
output/. A file is compared with its reference file by the module's own --check (the
step says how) or by this runner: byte-identical after LF normalization, or else token
by token, text and integers exactly and floating-point numbers to
|a - b| <= 1e-9 max(1, |b|) ('equal within 1e-9', with the largest relative
difference). numpy, LAPACK and BLAS builds and platforms can move the last digits of
floating-point results; 1e-9 is far below every printed precision.

--full regenerates every simulation of the section from the recorded seeds into
output/ (never into reference/ or inputs/) and compares each regenerated file with its
reference file as above. Some regenerations take hours: without --yes the script
prints a warning, the plan and the estimated times, and exits with status 2 without
running anything. The long modules keep resumable state (output/replicate/steps/, and
output/critical_values/replicate_full/ for Table 2), so an interrupted --full run
resumes where it stopped; remove those folders to start again from nothing.

Exit status: 0 if every step passes, 1 if one fails, 2 on a usage error or a --full
run without --yes. No PYTHONPATH is needed and the current directory does not matter:
every path is resolved from the package root.
"""

import argparse
import csv
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

from mlb import steps as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output", "replicate")
REL_TOL = S.REL_TOL
# a file a step compares must have been written while the step ran (seconds of slack
# for the resolution of file times)
MTIME_SLACK = 2.0
_NUM = re.compile(r"[-+]?(?:(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|\b(?:nan|inf)\b)")
_INT = re.compile(r"[-+]?\d+")
ALL_SCRIPT = "replicate_all.py"


def _p(rel):
    """Filesystem path of a package-relative path."""
    return os.path.join(ROOT, *rel.split("/"))


def fmt_s(s):
    """Seconds as a short duration."""
    if s is None:
        return "-"
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


# --------------------------------------------------------------------------------------
# comparison of a regenerated file with its reference file
# --------------------------------------------------------------------------------------


class _Diff:
    """The largest relative difference |a - b| / max(1, |b|) met in a comparison."""

    def __init__(self):
        self.max_rel = 0.0


def _close_num(x, y, diff):
    """Two floats equal to REL_TOL (relative to max(1, |y|)); NaN equals NaN."""
    if x == y:
        return True
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    if math.isinf(x) or math.isinf(y):
        return False
    rel = abs(x - y) / max(1.0, abs(y))
    diff.max_rel = max(diff.max_rel, rel)
    return rel <= REL_TOL


def _close_token(u, v, diff):
    """Two number tokens: integers exactly, otherwise as floats to REL_TOL."""
    if u == v:
        return True
    if _INT.fullmatch(u) and _INT.fullmatch(v):
        return int(u) == int(v)
    return _close_num(float(u), float(v), diff)


def compare_json(a, b, ignore, diff, where="$", parent=None):
    """First difference between two parsed JSON documents, or None. Keys in `ignore`
    are left out at every level, and a pair (parent, key) in `ignore` leaves out `key`
    inside the entry `parent`; integers and other values compare exactly, floats to
    REL_TOL."""
    if isinstance(a, dict) and isinstance(b, dict):
        def kept(d):
            return {k for k in d if k not in ignore and (parent, k) not in ignore}

        ka, kb = kept(a), kept(b)
        if ka != kb:
            return f"{where}: keys differ ({sorted(ka ^ kb)[:5]})"
        for k in sorted(ka):
            d = compare_json(a[k], b[k], ignore, diff, f"{where}.{k}", k)
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{where}: lengths {len(a)} and {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = compare_json(x, y, ignore, diff, f"{where}[{i}]", parent)
            if d:
                return d
        return None
    if isinstance(a, bool) or isinstance(b, bool):
        return None if a is b else f"{where}: {a!r} != {b!r}"
    if isinstance(a, int) and isinstance(b, int):
        return None if a == b else f"{where}: {a!r} != {b!r}"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        ok = _close_num(float(a), float(b), diff)
        return None if ok else f"{where}: {a!r} != {b!r}"
    return None if a == b else f"{where}: {a!r} != {b!r}"


def _cmp_line(x, y, diff):
    """None if two lines are equal token by token, else a description."""
    if x == y:
        return None
    nx, ny = _NUM.findall(x), _NUM.findall(y)
    if _NUM.sub("#", x) != _NUM.sub("#", y) or len(nx) != len(ny):
        return f"{x[:80]!r} != {y[:80]!r}"
    for u, v in zip(nx, ny):
        if not _close_token(u, v, diff):
            return f"{u} != {v}"
    return None


def _lines(text):
    return text[:-1].split("\n") if text.endswith("\n") else text.split("\n")


def _drop_columns(text, names):
    """A CSV text without the columns `names` (when its header has them)."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return text
    keep = [i for i, h in enumerate(rows[0]) if h not in names]
    if len(keep) == len(rows[0]):
        return text
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for r in rows:
        w.writerow([r[i] for i in keep if i < len(r)])
    return buf.getvalue()


def _cmp_text(ta, tb, diff, rows=None):
    """First difference between two texts, line by line and token by token; with
    rows (a steps.Rows), each line of `ta` against the line of `tb` with the same
    first rows.key_cols comma-separated fields, the header against the header."""
    la, lb = _lines(ta), _lines(tb)
    if rows is None:
        if len(la) != len(lb):
            return f"{len(la)} and {len(lb)} lines"
        for i, (x, y) in enumerate(zip(la, lb), 1):
            d = _cmp_line(x, y, diff)
            if d:
                return f"line {i}: {d}"
        return None
    d = _cmp_line(la[0], lb[0], diff)
    if d:
        return f"header: {d}"

    def key(line):
        return tuple(line.split(",")[:rows.key_cols])

    ref = {key(y): y for y in lb[1:]}
    mine = {key(x): x for x in la[1:]}
    missing = [k for k in rows.keys if tuple(k) not in mine]
    if missing:
        return f"rows not written: {missing}"
    if not mine:
        return "no row written"
    for k, x in mine.items():
        if k not in ref:
            return f"row {k} is not in the reference file"
        d = _cmp_line(x, ref[k], diff)
        if d:
            return f"row {k}: {d}"
    return None


def compare_files(new, ref, ignore_keys=(), rows=None, since=None):
    """Compare a regenerated file with its reference file: ('byte-identical' | 'equal
    within 1e-9' | 'DIFFERENT' | 'MISSING' | 'STALE', detail). Line endings are
    normalized to LF. Otherwise JSON documents are compared as data, without the keys
    in ignore_keys; a CSV without the columns named in ignore_keys; texts token by
    token (rows: see _cmp_text). 'equal within 1e-9' carries the largest relative
    difference met. since: the time the step started; a file written before it is
    STALE (left by an earlier run)."""
    if not os.path.exists(new):
        return "MISSING", f"not written: {os.path.relpath(new, ROOT)}"
    if not os.path.exists(ref):
        return "MISSING", f"no reference file: {os.path.relpath(ref, ROOT)}"
    if since is not None and os.path.getmtime(new) < since - MTIME_SLACK:
        return "STALE", f"not written by this run: {os.path.relpath(new, ROOT)}"
    with open(new, "rb") as fh:
        a = fh.read().replace(b"\r\n", b"\n")
    with open(ref, "rb") as fh:
        b = fh.read().replace(b"\r\n", b"\n")
    if a == b and rows is None:
        return "byte-identical", ""
    try:
        ta, tb = a.decode("utf-8"), b.decode("utf-8")
    except UnicodeDecodeError:
        return "DIFFERENT", "binary files differ"
    diff = _Diff()
    if ref.endswith(".json"):
        try:
            ja, jb = json.loads(ta), json.loads(tb)
        except ValueError:
            ja = jb = None
        if ja is not None:
            d = compare_json(ja, jb, set(ignore_keys), diff)
            if d is not None:
                return "DIFFERENT", d
            skipped = sorted(k for k in ignore_keys if isinstance(k, str)
                             and (k in ja or k in jb))
            note = f"largest relative difference {diff.max_rel:.1e}"
            if skipped:
                note += f"; apart from {', '.join(skipped)}"
            return "equal within 1e-9", note
    if ref.endswith(".csv") and ignore_keys:
        ta, tb = _drop_columns(ta, ignore_keys), _drop_columns(tb, ignore_keys)
    d = _cmp_text(ta, tb, diff, rows)
    if d is not None:
        return "DIFFERENT", d
    if rows is not None and diff.max_rel == 0.0 and all(
            x in _lines(tb) for x in _lines(ta)):
        return "byte-identical", "rows"
    return "equal within 1e-9", f"largest relative difference {diff.max_rel:.1e}"


def _summary(results):
    """One phrase for the comparisons of a step: counts and the largest difference."""
    n = len(results)
    ident = sum(v == "byte-identical" for _, v, _ in results)
    within = [(f, d) for f, v, d in results if v == "equal within 1e-9"]
    if not within:
        return f"{n} file(s), byte-identical"
    worst = max(within, key=lambda x: float(x[1].split()[3].rstrip(";")))
    return (f"{n} file(s): {ident} byte-identical, {len(within)} equal within 1e-9 "
            f"(largest relative difference {worst[1].split()[3].rstrip(';')}, "
            f"{os.path.basename(worst[0])})")


# --------------------------------------------------------------------------------------
# running steps
# --------------------------------------------------------------------------------------


def command(step, jobs):
    """The argument list of a step (script path relative to the package root)."""
    return [a.replace("{jobs}", str(jobs)) for a in step.argv]


def shown(step, jobs):
    return "python " + " ".join(command(step, jobs))


def run_step(step, jobs, failed_before):
    """Run one step; return a record dict (step, command, wall_s, rc, result, detail,
    comparisons, products)."""
    log_dir = os.path.join(OUT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(OUT, "steps", step.key), exist_ok=True)
    log = os.path.join(log_dir, f"{step.key}.log")
    rec = {"step": step.key, "command": shown(step, jobs), "wall_s": 0.0, "rc": "",
           "result": "FAIL", "detail": "", "comparisons": [], "products": [],
           "since": None}
    blocked = [n for n in step.needs if n in failed_before]
    if blocked:
        rec["detail"] = f"not run: {', '.join(blocked)} failed"
        return rec
    for rel in step.fresh:
        path = _p(rel)
        if not rel.startswith("output/replicate/steps/"):
            raise ValueError(f"{step.key}: refusing to remove {rel}")
        if os.path.isdir(path):
            shutil.rmtree(path)
    for src, dst in step.setup:
        os.makedirs(os.path.dirname(_p(dst)), exist_ok=True)
        shutil.copyfile(_p(src), _p(dst))
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    since = time.time()
    rec["since"] = since
    t0 = time.perf_counter()
    with open(log, "wb") as fh:
        p = subprocess.run([sys.executable, *command(step, jobs)], cwd=ROOT, env=env,
                           stdout=fh, stderr=subprocess.STDOUT, check=False)
    rec["wall_s"] = round(time.perf_counter() - t0, 1)
    rec["rc"] = p.returncode
    if p.returncode != 0 and p.returncode not in step.info_rc:
        rec["detail"] = f"exit status {p.returncode}"
        return rec
    results, bad = [], []
    for c in step.compare:
        new, ref = c[0], c[1]
        rows = c[2] if len(c) > 2 else None
        verdict, detail = compare_files(_p(new), _p(ref), step.ignore_keys, rows,
                                        since)
        results.append((new, verdict, detail))
        rec["comparisons"].append((new, ref, verdict, detail))
        if verdict not in ("byte-identical", "equal within 1e-9"):
            bad.append(f"{ref}: {verdict} {detail}".strip())
    if bad:
        rec["detail"] = "; ".join(bad)
        return rec
    rec["result"] = "PASS"
    parts = []
    if results:
        parts.append(_summary(results))
    if step.check:
        verb = "compared" if step.source == "recomputed" else "checked"
        parts.append(f"{verb} by {step.check.split(':')[0]}")
    if p.returncode != 0:
        parts.append(f"the module's own byte comparison exited {p.returncode} "
                     "(information: the comparison above decides)")
    rec["detail"] = "; ".join(parts)
    rec["products"] = [x for x in step.products if os.path.exists(_p(x))]
    return rec


def verdict_line(step, rec):
    """The result of a step in words."""
    if rec["result"] != "PASS":
        return f"DIFFERENT or FAILED: {rec['detail']}"
    if step.source == "reference":
        detail = f"; {rec['detail']}" if rec["detail"] else ""
        return f"shown from reference/ (regenerated by {step.regenerated_by}){detail}"
    return f"recomputed: equal to reference/ ({rec['detail']})"


def _tail(path, n=15):
    """The last n lines of a file (every line when n is None)."""
    try:
        with open(path, "rb") as fh:
            lines = fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []
    return lines if n is None else lines[-n:]


def run_steps(keys, jobs, out=sys.stdout):
    """Run the steps `keys` in order; return their records."""
    recs, failed = [], set()
    for i, k in enumerate(keys, 1):
        st = S.STEPS[k]
        out.write(f"[{i}/{len(keys)}] {k}: {st.title}\n")
        out.write(f"    $ {shown(st, jobs)}\n")
        out.flush()
        rec = run_step(st, jobs, failed)
        recs.append(rec)
        if rec["result"] != "PASS":
            failed.add(k)
        out.write(f"    {verdict_line(st, rec)}  [{fmt_s(rec['wall_s'])}]\n")
        if st.check:
            out.write(f"      how: {st.check}\n")
        for x in rec["products"]:
            out.write(f"      wrote: {x}\n")
        log = os.path.join(OUT, "logs", f"{k}.log")
        if st.echo:
            for ln in _tail(log, None):
                if ln.startswith(st.echo):
                    out.write(f"      info: {ln}\n")
        if rec["result"] != "PASS" and rec["rc"] not in ("", 0):
            for ln in _tail(log):
                out.write(f"      | {ln}\n")
        out.flush()
    return recs


# --------------------------------------------------------------------------------------
# the section scripts
# --------------------------------------------------------------------------------------


SECTION_SCRIPTS = ("replicate_sec_2_3", "replicate_sec_4", "replicate_sec_5",
                   "replicate_sec_6", "replicate_sup_S1", "replicate_sup_S2",
                   "replicate_sup_S3", "replicate_sup_S4", "replicate_sup_S5")


def all_keys(mode):
    """The step keys of the nine section scripts, in their order, each once (mode
    'quick' or 'full')."""
    import importlib
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    out = []
    for name in SECTION_SCRIPTS:
        mod = importlib.import_module(name)
        for k in (mod.FULL if mode == "full" else mod.QUICK):
            if k not in out:
                out.append(k)
    return out


def warning(name, title, keys, jobs, quick_keys=()):
    """The text printed before a --full run (and instead of it without --yes)."""
    pl = S.plan(keys)
    total = S.seconds(keys)
    every = S.seconds(all_keys("full"))
    name_of = title.split(":")[0]
    if S.plan(keys) == S.plan(quick_keys):
        lines = [f"{name_of} has no long simulation: --full runs the same steps as the "
                 "quick mode."]
    else:
        lines = [f"--full regenerates the simulations of {name_of} from the recorded "
                 "seeds and compares them with reference/; the results it does not "
                 "regenerate are shown from reference/, with the script that "
                 "regenerates them."]
    if name == ALL_SCRIPT:
        lines.append(f"Estimated time: {fmt_s(total)} at 8 worker processes (every "
                     "section, each step once).")
    else:
        lines.append(f"Estimated time: {fmt_s(total)} at 8 worker processes (all "
                     f"sections, each step once: {fmt_s(every)} with "
                     f"`python {ALL_SCRIPT} --full`).")
    lines.append("The time grows roughly in proportion as the number of worker "
                 f"processes falls (--jobs; this run uses {jobs}, and this machine has "
                 f"{os.cpu_count() or 1} logical processors).")
    if "full_overdetrend" in pl:
        lines.append("One step was never run whole on this version: full_overdetrend "
                     "(428,000,000 draws); its "
                     f"{fmt_s(S.STEPS['full_overdetrend'].seconds)} is an estimate "
                     "from CPU time.")
    lines.append("The quick mode (no option) is enough to check the paper: it "
                 "recomputes the exact results and the small simulations and reads the "
                 "long simulations from reference/.")
    return lines


def parse(argv, prog, doc):
    ap = argparse.ArgumentParser(prog=prog, description=doc,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="quick mode, the default (minutes)")
    ap.add_argument("--full", action="store_true",
                    help="regenerate the simulations from the recorded seeds (hours): "
                    "prints a warning and the plan, and starts only with --yes")
    ap.add_argument("--yes", action="store_true", help="start a --full run")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1),
                    help="worker processes (default: min(8, number of logical "
                    "processors))")
    a = ap.parse_args(argv)
    if a.quick and a.full:
        ap.error("--quick and --full are exclusive; --quick is the default mode")
    if a.yes and not a.full:
        ap.error("--yes starts a --full run; it means nothing without --full")
    if a.jobs < 1:
        ap.error("--jobs must be at least 1")
    return a


def main(script, title, exhibits, quick, full, argv=None, doc=None):
    """The command line of a section script. script: its file name; title: the
    section; exhibits: (name, what) pairs; quick, full: the step keys of the two
    modes."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    a = parse(sys.argv[1:] if argv is None else argv, script, doc)
    keys = S.plan(full if a.full else quick)
    mode = "full" if a.full else "quick"
    print(f"{script}: {title}; {mode} mode, {a.jobs} worker process(es), "
          f"{len(keys)} step(s)")
    for name, what in exhibits:
        print(f"  {name}: {what}")
    if a.full:
        print()
        for ln in warning(script, title, full, a.jobs, quick):
            print(ln)
        print("Plan (estimated wall times at 8 worker processes):")
        for k in keys:
            print(f"  {fmt_s(S.STEPS[k].seconds):>9}  {k}: {shown(S.STEPS[k], a.jobs)}")
        if not a.yes:
            print("Nothing was run. Add --yes to start.")
            return 2
    print()
    t0 = time.perf_counter()
    recs = run_steps(keys, a.jobs)
    wall = time.perf_counter() - t0
    n_fail = sum(r["result"] != "PASS" for r in recs)
    print(f"\n{'step':<26} {'result':<6} {'time':>9}")
    for r in recs:
        src = S.STEPS[r["step"]].source
        print(f"{r['step']:<26} {r['result']:<6} {fmt_s(r['wall_s']):>9}  "
              f"{'recomputed' if src == 'recomputed' else 'shown from reference/'}")
    print(f"{len(recs)} step(s), {n_fail} failed; wall time {fmt_s(wall)}; "
          f"{'PASS' if n_fail == 0 else 'FAIL'}")
    write_steps_csv(os.path.join(OUT, f"{os.path.splitext(script)[0]}_{mode}.csv"),
                    recs)
    if not a.full and S.plan(full) != S.plan(quick):
        print(f"The long simulations were read from reference/; `python {script} "
              f"--full` regenerates them from the recorded seeds (estimated "
              f"{fmt_s(S.seconds(full))} at 8 worker processes).")
    elif not a.full:
        print("This section has no long simulation: --full runs the same steps.")
    return 1 if n_fail else 0


def write_steps_csv(path, recs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=("step", "command", "wall_s", "rc", "result",
                                           "detail"),
                           lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)
