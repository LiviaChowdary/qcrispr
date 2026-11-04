"""qcrispr: guide-RNA selection as a QUBO, solved classically and with QAOA.

The point of this package is a fair comparison, not a quantum result. It builds
a genuinely quadratic guide-selection problem out of off-target predictions,
then solves the same problem three ways -- exactly, with a classical heuristic,
and with QAOA on an exact statevector simulation -- and reports which one wins
on solution quality and on time.
"""

__version__ = "0.1.0"
