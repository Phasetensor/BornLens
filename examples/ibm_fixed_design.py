"""One prespecified hardware comparison. Pass an existing job ID to resume."""

import argparse
import json
from dataclasses import asdict
from math import pi

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from bornlens import compare
from bornlens.qiskit import from_counts


def run(service, backend_name, job_id=None):
    backend = service.backend(backend_name)
    if backend.configuration().simulator:
        raise ValueError("select a physical IBM backend")
    if job_id:
        job = service.job(job_id)
        if job.backend().name != backend_name or "bornlens-fixed-z-v1" not in job.tags:
            raise ValueError("job does not match this protocol")
    else:
        circuits = []
        for angle in (0, pi / 3):
            circuit = QuantumCircuit(1)
            circuit.ry(angle, 0)
            circuit.measure_all()
            circuits.append(circuit)
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=1,
            seed_transpiler=47,
            initial_layout=[0],
        )
        sampler = SamplerV2(mode=backend)
        sampler.options.environment.job_tags = ["bornlens-fixed-z-v1"]
        job = sampler.run(pm.run(circuits), shots=4096)
    print("Job:", job.job_id(), flush=True)
    a, b = [pub.data.meas.get_counts() for pub in job.result()]
    if any(sum(counts.values()) != 4096 for counts in (a, b)):
        raise ValueError("unexpected shot count")
    result = compare(from_counts(a, "Z"), from_counts(b, "Z"), tolerance=0.1)
    return dict(
        backend=backend_name, job_id=job.job_id(), counts=[a, b], result=asdict(result)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--job-id")
    args = parser.parse_args()
    print(json.dumps(run(QiskitRuntimeService(), args.backend, args.job_id), indent=2))
