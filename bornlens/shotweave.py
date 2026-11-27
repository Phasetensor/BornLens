"""Read completed ShotWeave hardware JSONL exports for single-qubit Z."""
import json
from . import Sample, _integer, compare


def compare_final(path, arm_a, arm_b, **settings):
    """Condition on the pilot allocation and use only fresh final-stage outcomes."""
    with open(path, encoding="utf8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    final, = [r for r in rows if "total_shots" in r]
    job, = [r for r in rows if r.get("stage") == "final" and "job_id" in r]
    pilot, = [r for r in rows if r.get("stage", "pilot") == "pilot" and "job_id" in r]
    shots, ones = final["final_shots"], final["successes"]
    if (pilot["job_id"] == job["job_id"] or job["shots"] != shots or
        len(shots) != len(ones) or len(pilot["shots"]) != len(shots) or
        sum(shots)+sum(pilot["shots"]) != final["total_shots"]):
        raise ValueError("inconsistent ShotWeave stages or shot accounting")
    samples = [Sample("Z", _integer(n,"shots",1)-_integer(k,"ones"), k)
               for n,k in zip(shots, ones)]
    indices = [_integer(i,"arm") for i in (arm_a, arm_b)]
    if arm_a == arm_b or max(indices) >= len(samples):
        raise ValueError("choose two distinct valid arms")
    return compare(*(samples[i] for i in indices), **settings)
