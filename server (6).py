"""Ghost Data Auditor: dependency-free web server that connects the frontend to your backend.

Run from PyCharm (right-click > Run) or:  python frontend/server.py [options]

  --csv      results file to show            (default: ../output.csv)
  --backend  folder holding code/ and dataset/ from your unzipped code.zip.
             Optional. When given, the dashboard gets a "Re-run audit" button that runs
             <backend>/code/main.py exactly as its README describes. Backend files are never edited.
  --port     default 8000
"""
import argparse, csv, json, os, shutil, subprocess, sys, threading, time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=os.path.join(HERE, "..", "output.csv"))
ap.add_argument("--backend", default=None)
ap.add_argument("--dataset", default=None, help="dataset folder (default: <backend>/dataset)")
ap.add_argument("--port", type=int, default=8000)
ARGS = ap.parse_args()

RUN_DIR = os.path.join(HERE, "runs")          # re-runs write here, never over your submission files
STATE = {"running": False, "message": "", "finished": None, "ok": None, "log": ""}
LOCK = threading.Lock()


def current_csv():
    latest = os.path.join(RUN_DIR, "output.csv")
    return latest if os.path.isfile(latest) else ARGS.csv


def load_cases():
    cases = []
    with open(current_csv(), newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                steps = json.loads(row["steps_json"] or "[]")
            except json.JSONDecodeError:
                steps = []
            cases.append({"case_id": row["case_id"], "status": row["deletion_status"], "steps": steps})
    return sorted(cases, key=lambda c: c["case_id"])


def run_report():
    for folder in (RUN_DIR, os.path.dirname(os.path.abspath(ARGS.csv))):
        p = os.path.join(folder, "run_report.json")
        if os.path.isfile(p):
            try:
                d = json.load(open(p, encoding="utf-8-sig"))
                return {k: d.get(k) for k in ("requests", "total_seconds", "scan_files", "model_calls")}
            except (OSError, ValueError):
                pass
    return None


def run_backend():
    main_py = os.path.join(ARGS.backend, "code", "main.py")
    dataset = os.path.join(ARGS.backend, "dataset")
    try:
        os.makedirs(RUN_DIR, exist_ok=True)
        for old in ("output.csv", "partial_output.csv"):
            if os.path.exists(os.path.join(RUN_DIR, old)):
                os.remove(os.path.join(RUN_DIR, old))
        p = subprocess.run([sys.executable, main_py, "--dataset", dataset, "--output-dir", RUN_DIR],
                           cwd=ARGS.backend, capture_output=True, text=True)
        ok = p.returncode == 0 and os.path.isfile(os.path.join(RUN_DIR, "output.csv"))
        STATE.update(ok=ok, log=(p.stdout + p.stderr)[-1500:],
                     message="Audit finished. Results updated." if ok else "Audit did not finish cleanly. Showing the previous results.")
    except Exception as e:
        STATE.update(ok=False, message=f"Could not start the backend: {e}", log=str(e))
    finally:
        STATE.update(running=False, finished=time.strftime("%H:%M:%S"))


# ---------- person lookup (reads the dataset; never writes to it) ----------
_IDX = {}


def dataset_dir():
    return ARGS.dataset or (os.path.join(ARGS.backend, "dataset") if ARGS.backend else None)


def _rows(name):
    with open(os.path.join(dataset_dir(), name), newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _norm(v):
    v = (v or "").strip()
    return v.lower() if "@" in v else (("".join(ch for ch in v if ch.isdigit()) if sum(ch.isdigit() for ch in v) >= 7 else v.lower()))


def build_index():
    users = {r["user_id"]: r for r in _rows("user_identities.csv")}
    ident, hist = {}, {}
    for uid, r in users.items():
        for k in ("email", "phone", "name"):
            ident.setdefault(_norm(r[k]), {}).setdefault(uid, "current " + k)
    for a in _rows("user_audit_log.csv"):
        if a["field_name"] in ("email", "phone"):
            hist.setdefault(a["user_id"], []).append({"field": a["field_name"], "old": a["old_value"], "new": a["new_value"], "when": a["event_time"][:10]})
            for v in (a["old_value"], a["new_value"]):
                ident.setdefault(_norm(v), {}).setdefault(a["user_id"], "previous " + a["field_name"])
    reqs = _rows("deletion_requests.csv")
    _IDX.update(users=users, ident=ident, hist=hist, reqs=reqs)


def lookup(q):
    if not _IDX:
        build_index()
    users, ident, hist, reqs = _IDX["users"], _IDX["ident"], _IDX["hist"], _IDX["reqs"]
    nq = _norm(q)
    found = {}
    if nq in ident:
        found = dict(ident[nq])
    elif len(q.strip()) >= 3 and "@" not in q and not nq.isdigit():
        for uid, r in users.items():
            if nq in r["name"].lower() and len(found) < 5:
                found[uid] = "name"
    status = {c["case_id"]: c["status"] for c in load_cases()}
    people = []
    for uid, how in list(found.items())[:5]:
        u = users[uid]
        ids = {_norm(u["email"])} | {_norm(h[k]) for h in hist.get(uid, []) for k in ("old", "new") if h["field"] == "email"}
        cases = []
        for r in reqs:
            e = _norm(r["email"])
            if e in ids:
                cases.append({"case_id": r["case_id"], "status": status.get(r["case_id"], "UNKNOWN"), "reported": r["reported_status"],
                              "requested": r["requested_at"][:10], "via": "current email" if e == _norm(u["email"]) else "previous email"})
        people.append({"user_id": uid, "name": u["name"], "email": u["email"], "phone": u["phone"], "matched": how,
                       "previous": hist.get(uid, []), "cases": sorted(cases, key=lambda c: c["case_id"])})
    return people


def examples():
    if not _IDX:
        build_index()
    users, reqs = _IDX["users"], _IDX["reqs"]
    cur = {_norm(r["email"]) for r in users.values()}
    status = {c["case_id"]: c["status"] for c in load_cases()}
    out, seen = [], set()
    for label, test in (("Searched by an old email address", lambda r: _norm(r["email"]) not in cur),
                        ("Company said complete, audit disagrees", lambda r: r["reported_status"] == "COMPLETE" and status.get(r["case_id"]) == "INCOMPLETE"),
                        ("Company said not started", lambda r: r["reported_status"] == "NOT_STARTED")):
        r = next((x for x in reqs if test(x) and x["email"] not in seen), None)
        if r:
            seen.add(r["email"]); out.append({"label": label, "q": r["email"]})
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if urlparse(self.path).path != "/api/rerun":
            return self._send(404, {"error": "not found"})
        if not ARGS.backend:
            return self._send(400, {"error": "Start the server with --backend <folder> to enable re-runs."})
        with LOCK:
            if STATE["running"]:
                return self._send(409, {"error": "An audit is already running."})
            STATE.update(running=True, message="Audit running. This can take a few minutes.", ok=None)
        threading.Thread(target=run_backend, daemon=True).start()
        self._send(202, STATE)

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/cases":
                return self._send(200, load_cases())
            if path == "/api/summary":
                cs = load_cases()
                return self._send(200, {"total": len(cs), "by_status": Counter(c["status"] for c in cs)})
            if path.startswith("/api/cases/"):
                cid = path.rsplit("/", 1)[1]
                hit = next((c for c in load_cases() if c["case_id"] == cid), None)
                return self._send(200, hit) if hit else self._send(404, {"error": "not found"})
            if path in ("/api/lookup", "/api/examples"):
                if not dataset_dir() or not os.path.isdir(dataset_dir()):
                    return self._send(409, {"error": "Lookup needs the dataset. Start the server with --backend <folder> (the folder that contains dataset/)."})
                if path == "/api/examples":
                    return self._send(200, examples())
                q = parse_qs(urlparse(self.path).query).get("q", [""])[0].strip()
                return self._send(200, {"q": q, "people": lookup(q) if q else []})
            if path == "/api/status":
                return self._send(200, {**STATE, "can_rerun": bool(ARGS.backend), "source": os.path.basename(current_csv()),
                                        "from_rerun": current_csv().startswith(RUN_DIR), "report": run_report()})
        except FileNotFoundError as e:
            return self._send(500, {"error": f"{e.filename or current_csv()} not found"})
        pages = {"/": "landing.html", "/dashboard": "index.html", "/index.html": "index.html"}
        if path in pages:
            with open(os.path.join(HERE, pages[path]), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        self._send(404, {"error": "not found"})


if __name__ == "__main__":
    print(f"Results file : {os.path.abspath(current_csv())}")
    print(f"Backend      : {os.path.abspath(ARGS.backend) if ARGS.backend else 'not connected (read-only mode)'}")
    print(f"Open         : http://localhost:{ARGS.port}")
    ThreadingHTTPServer(("", ARGS.port), Handler).serve_forever()
