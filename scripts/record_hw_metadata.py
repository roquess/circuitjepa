"""Record IBM hardware metadata: per-job usage/timestamps and a backend calibration snapshot.

Reproduces the two metadata files used by the paper:

- results/hardware/job_metrics.json: for every job id in results/hardware/jobs.json,
  the job's usage (QPU charge time) and timestamps, plus the account usage after the jobs;
- results/hardware/calibration_<backend>.json: `backend.properties().to_dict()`, i.e.
  the calibration snapshot the device reports at the time this script runs (it is not
  the calibration in force when the jobs ran; see `last_update_date` and the per-value dates).

It needs IBM credentials saved with QiskitRuntimeService.save_account(...). It only reads
job and backend information; it submits nothing and spends no QPU time.

Usage:
    python scripts/record_hw_metadata.py                 # all jobs in jobs.json, their backend
    python scripts/record_hw_metadata.py --jobs JOB_ID ... --backend ibm_marrakesh
    python scripts/record_hw_metadata.py --no-calibration --out-dir /tmp/meta
"""

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

from circuitjepa import experiments as ex


def _plain(obj):
    """JSON-serialisable copy (datetimes as ISO strings)."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def job_record(job) -> dict:
    metrics = job.metrics()
    return {
        "job_id": job.job_id(),
        "backend": job.backend().name,
        "usage": _plain(metrics.get("usage")),
        "timestamps": _plain(metrics.get("timestamps")),
        "qiskit_version": metrics.get("qiskit_version"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", nargs="+", help="job ids (default: every job in results/hardware/jobs.json)")
    ap.add_argument("--backend", help="backend for the calibration snapshot (default: the jobs' backend)")
    ap.add_argument("--no-calibration", action="store_true", help="skip the calibration snapshot")
    ap.add_argument("--out-dir", type=Path, help="output directory (default: results/hardware)")
    args = ap.parse_args(argv)

    from qiskit_ibm_runtime import QiskitRuntimeService

    hw = ex.results_dir() / "hardware"
    out = args.out_dir or hw
    out.mkdir(parents=True, exist_ok=True)
    tags = {}
    if args.jobs:
        job_ids = args.jobs
    else:
        recs = [json.loads(line) for line in (hw / "jobs.json").read_text(encoding="utf-8").splitlines() if line.strip()]
        job_ids = [r["job_id"] for r in recs]
        tags = {r["job_id"]: r["tag"] for r in recs}

    try:
        service = QiskitRuntimeService()
    except Exception as exc:
        print(f"error: could not connect to IBM Quantum ({exc}); save credentials first", file=sys.stderr)
        return 1

    metrics, backends = {}, set()
    for jid in job_ids:
        rec = job_record(service.job(jid))
        metrics[tags.get(jid, jid)] = rec
        backends.add(rec["backend"])
        print(f"{tags.get(jid, jid)}: {rec['usage']}")
    metrics["account_usage_after"] = _plain(service.usage())
    ex.write_json(out / "job_metrics.json", metrics)
    print(f"wrote {out / 'job_metrics.json'}")

    if not args.no_calibration:
        name = args.backend or (backends.pop() if len(backends) == 1 else None)
        if name is None:
            print("error: jobs span several backends; pass --backend", file=sys.stderr)
            return 1
        props = service.backend(name).properties()
        ex.write_json(out / f"calibration_{name}.json", _plain(props.to_dict()), indent=None)
        print(f"wrote {out / f'calibration_{name}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
