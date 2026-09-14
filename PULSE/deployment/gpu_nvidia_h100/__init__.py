"""H100 GPU deployment with CPU entropy coding."""
import os

HOST_CPUS = frozenset(os.sched_getaffinity(0))

for name, value in (("OMP_PROC_BIND", "close"), ("OMP_PLACES", "cores"),
                    ("OMP_DYNAMIC", "false"), ("OMP_WAIT_POLICY", "PASSIVE")):
    os.environ.setdefault(name, value)
