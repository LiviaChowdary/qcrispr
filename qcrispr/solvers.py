"""Three solvers for the same QUBO: exact, classical heuristic, and QAOA.

The QAOA implementation is an exact statevector simulation written directly in
numpy. No quantum SDK is involved, and nothing here runs on quantum hardware.
That is not a shortcut: a QUBO cost Hamiltonian is diagonal in the computational
basis, so the cost layer is a phase applied per basis state and the mixer is a
product of single-qubit rotations. Simulating it exactly is a handful of array
operations, and the result is identical to what a noiseless simulator in any
framework would return.

The cost is memory: an exact statevector needs 2^n complex amplitudes, so this
tops out around 18-20 variables on a laptop. Real quantum hardware would remove
that specific ceiling and replace it with noise, limited qubit counts, and
limited circuit depth.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .qubo import QUBO

BRUTE_FORCE_MAX_N = 24
QAOA_MAX_N = 18


@dataclass
class SolverResult:
    name: str
    x: np.ndarray
    energy: float
    seconds: float
    extra: dict = field(default_factory=dict)

    @property
    def support(self) -> list[int]:
        return [int(i) for i in np.flatnonzero(self.x)]


def _states_bits(n: int) -> np.ndarray:
    states = np.arange(1 << n, dtype=np.int64)
    return ((states[:, None] >> np.arange(n, dtype=np.int64)) & 1).astype(np.float64)


def cost_diagonal(qubo: QUBO) -> np.ndarray:
    """Energy of every basis state, indexed so that bit i of the index is x_i."""
    bits = _states_bits(qubo.n)
    return ((bits @ qubo.Q) * bits).sum(axis=1) + qubo.offset


def brute_force(qubo: QUBO, max_n: int = BRUTE_FORCE_MAX_N) -> SolverResult:
    """Exact optimum by enumeration: the ground truth other solvers are scored against."""
    if qubo.n > max_n:
        raise ValueError(f"brute force refuses n={qubo.n} (limit {max_n})")
    t0 = time.perf_counter()
    diag = cost_diagonal(qubo)
    best = int(np.argmin(diag))
    x = np.array([(best >> i) & 1 for i in range(qubo.n)], dtype=np.int64)
    return SolverResult(
        "brute_force",
        x,
        float(diag[best]),
        time.perf_counter() - t0,
        {"evaluated": int(diag.size)},
    )


def simulated_annealing(
    qubo: QUBO,
    restarts: int = 20,
    sweeps: int = 400,
    t_start: float = 2.0,
    t_end: float = 0.01,
    seed: int = 0,
) -> SolverResult:
    """Single-flip Metropolis with geometric cooling and random restarts."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = qubo.n
    Q = qubo.Q
    best_x = None
    best_e = np.inf
    temps = t_start * (t_end / t_start) ** (np.arange(sweeps) / max(1, sweeps - 1))

    for _ in range(restarts):
        x = rng.integers(0, 2, size=n).astype(np.float64)
        e = float(x @ Q @ x + qubo.offset)
        for temp in temps:
            for i in rng.permutation(n):
                # delta from flipping x_i, using the symmetric-matrix convention
                s = 1.0 - 2.0 * x[i]
                delta = s * (Q[i, i] + 2.0 * (Q[i] @ x - Q[i, i] * x[i]))
                if delta <= 0 or rng.random() < np.exp(-delta / temp):
                    x[i] += s
                    e += delta
        if e < best_e:
            best_e = e
            best_x = x.copy()

    x_int = best_x.astype(np.int64)
    return SolverResult(
        "simulated_annealing",
        x_int,
        qubo.energy(x_int),
        time.perf_counter() - t0,
        {"restarts": restarts, "sweeps": sweeps},
    )


def feasible_spectrum(qubo: QUBO, k: int) -> np.ndarray:
    """Sorted energies of every weight-k assignment.

    Used to report instance hardness. A solver result is uninterpretable
    without it: if greedy already ranks first out of a few hundred feasible
    sets, then nothing else about the instance is informative, and any claim
    that some method "found the optimum" says more about the instance than the
    method.
    """
    diag = cost_diagonal(qubo)
    weights = _popcount(np.arange(1 << qubo.n, dtype=np.int64))
    return np.sort(diag[weights == k])


def rank_of(energy: float, spectrum: np.ndarray, tol: float = 1e-9) -> int:
    """1-based rank of `energy` within `spectrum` (ties take the best rank)."""
    return int(np.searchsorted(spectrum, energy - tol) + 1)


def greedy(qubo: QUBO, k: int) -> SolverResult:
    """Add the single best variable k times, scoring against the full objective.

    Worth running on every instance for one reason: it says whether the
    instance is hard at all. If greedy already finds the optimum, no result
    about any other solver on that instance means anything.
    """
    t0 = time.perf_counter()
    n = qubo.n
    x = np.zeros(n, dtype=np.float64)
    chosen: list[int] = []
    for _ in range(k):
        best_i, best_e = None, np.inf
        for i in range(n):
            if x[i]:
                continue
            x[i] = 1.0
            e = float(x @ qubo.Q @ x)
            x[i] = 0.0
            if e < best_e:
                best_i, best_e = i, e
        x[best_i] = 1.0
        chosen.append(best_i)
    x_int = x.astype(np.int64)
    return SolverResult(
        "greedy", x_int, qubo.energy(x_int), time.perf_counter() - t0,
        {"order": chosen},
    )


def repair_to_cardinality(x: np.ndarray, qubo: QUBO, k: int) -> np.ndarray:
    """Greedily add or drop variables until exactly k are set.

    A penalty-encoded solver can return an infeasible bitstring, and throwing
    that away understates it. Cheap repair is what a practitioner would
    actually do, so the penalty encoding is compared with repair applied rather
    than against its raw infeasible output.
    """
    work = np.asarray(x, dtype=np.float64).copy()
    while int(work.sum()) > k:
        candidates = np.flatnonzero(work)
        best_i, best_e = None, np.inf
        for i in candidates:
            work[i] = 0.0
            e = float(work @ qubo.Q @ work)
            work[i] = 1.0
            if e < best_e:
                best_i, best_e = i, e
        work[best_i] = 0.0
    while int(work.sum()) < k:
        candidates = np.flatnonzero(work == 0)
        best_i, best_e = None, np.inf
        for i in candidates:
            work[i] = 1.0
            e = float(work @ qubo.Q @ work)
            work[i] = 0.0
            if e < best_e:
                best_i, best_e = i, e
        work[best_i] = 1.0
    return work.astype(np.int64)


def constrained_annealing(
    qubo: QUBO,
    k: int,
    restarts: int = 20,
    sweeps: int = 400,
    t_start: float = 1.0,
    t_end: float = 0.005,
    seed: int = 0,
) -> SolverResult:
    """Simulated annealing over a swap neighbourhood, so every state is feasible.

    This is the classical counterpart of the XY mixer, and the baseline the
    quantum arm has to beat. Annealing on the penalised objective with
    single-flip moves is a much weaker method, because every move that changes
    the cardinality is fought by a large penalty, and the walk spends its time
    climbing in and out of that wall rather than exploring feasible sets.
    Swapping one selected variable for one unselected one keeps the Hamming
    weight fixed by construction, exactly as the Dicke-plus-XY route does.

    Pass the penalty-free objective QUBO: the constraint lives in the move set.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = qubo.n
    Q = qubo.Q
    if not 0 < k <= n:
        raise ValueError(f"k must be in 1..{n}, got {k}")

    temps = t_start * (t_end / t_start) ** (np.arange(sweeps) / max(1, sweeps - 1))
    best_x, best_e = None, np.inf

    for _ in range(restarts):
        x = np.zeros(n, dtype=np.float64)
        x[rng.choice(n, size=k, replace=False)] = 1.0
        qx = Q @ x
        e = float(x @ qx + qubo.offset)
        for temp in temps:
            selected = np.flatnonzero(x)
            unselected = np.flatnonzero(x == 0)
            if unselected.size == 0:
                break
            for _ in range(n):
                i = int(rng.choice(selected))
                j = int(rng.choice(unselected))
                # x' = x - e_i + e_j, with Q symmetric
                delta = (
                    2.0 * (qx[j] - qx[i])
                    + Q[i, i] + Q[j, j] - 2.0 * Q[i, j]
                )
                if delta <= 0 or rng.random() < np.exp(-delta / temp):
                    x[i], x[j] = 0.0, 1.0
                    qx += Q[:, j] - Q[:, i]
                    e += delta
                    selected = np.flatnonzero(x)
                    unselected = np.flatnonzero(x == 0)
        if e < best_e:
            best_e, best_x = e, x.copy()

    x_int = best_x.astype(np.int64)
    return SolverResult(
        "constrained_annealing",
        x_int,
        qubo.energy(x_int),
        time.perf_counter() - t0,
        {"restarts": restarts, "sweeps": sweeps, "neighbourhood": "swap"},
    )


def _apply_mixer(state: np.ndarray, n: int, beta: float) -> None:
    """Transverse-field mixer: RX(2*beta) on every qubit."""
    c = np.cos(beta)
    s = np.sin(beta)
    for q in range(n):
        v = state.reshape(-1, 2, 1 << q)
        a = v[:, 0, :].copy()
        b = v[:, 1, :].copy()
        v[:, 0, :] = c * a - 1j * s * b
        v[:, 1, :] = -1j * s * a + c * b


def _popcount(values: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values)
    counts = np.zeros_like(values)
    work = values.copy()
    while np.any(work):
        counts += work & 1
        work >>= 1
    return counts


def dicke_state(n: int, k: int) -> np.ndarray:
    """|D^n_k>: the uniform superposition over all weight-k bitstrings.

    Built directly in the statevector rather than through a preparation
    circuit. The amplitudes are what matter here, and a circuit would only add
    gate-count detail that an exact simulation does not see.
    """
    weights = _popcount(np.arange(1 << n, dtype=np.int64))
    mask = weights == k
    count = int(mask.sum())
    if count == 0:
        raise ValueError(f"no weight-{k} states for n={n}")
    state = np.zeros(1 << n, dtype=np.complex128)
    state[mask] = 1.0 / np.sqrt(count)
    return state


def _apply_xy_pair(state: np.ndarray, i: int, j: int, beta: float) -> None:
    """exp(-i*beta*(XX+YY)/2) on qubits i, j: a rotation within {|10>, |01>}.

    |00> and |11> are untouched, so this term conserves Hamming weight exactly.
    """
    c = np.cos(beta)
    s = np.sin(beta)
    bit_i, bit_j = 1 << i, 1 << j
    idx = np.arange(state.size, dtype=np.int64)
    sel_10 = idx[((idx & bit_i) != 0) & ((idx & bit_j) == 0)]
    sel_01 = sel_10 - bit_i + bit_j
    a = state[sel_10].copy()
    b = state[sel_01].copy()
    state[sel_10] = c * a - 1j * s * b
    state[sel_01] = -1j * s * a + c * b


def _apply_xy_ring(state: np.ndarray, n: int, beta: float) -> None:
    """Ring XY mixer, first-order Trotterised into even then odd pairs.

    Pairs within each set act on disjoint qubits and so commute exactly; the
    even/odd split is the usual decomposition. Trotter error affects how well
    this approximates exp of the full ring sum, but not the property that
    matters: every pair term conserves Hamming weight, so any product of them
    does too. A Dicke start therefore stays inside the exactly-k subspace for
    any beta, depth, or Trotter ordering.
    """
    if n < 2:
        return
    for parity in (0, 1):
        for i in range(parity, n, 2):
            j = (i + 1) % n
            if i != j:
                _apply_xy_pair(state, i, j, beta)


def qaoa_state(
    params: np.ndarray,
    diag: np.ndarray,
    n: int,
    mixer: str = "x",
    k: int | None = None,
    phase_scale: float = 1.0,
) -> np.ndarray:
    """Run the p-layer QAOA circuit and return the final statevector."""
    p = len(params) // 2
    gammas, betas = params[:p], params[p:]
    if mixer == "xy":
        if k is None:
            raise ValueError("the xy mixer needs k to prepare the Dicke state")
        state = dicke_state(n, k)
    else:
        state = np.full(1 << n, 1.0 / np.sqrt(1 << n), dtype=np.complex128)
    scaled = diag / phase_scale
    for layer in range(p):
        state *= np.exp(-1j * gammas[layer] * scaled)
        if mixer == "xy":
            _apply_xy_ring(state, n, betas[layer])
        else:
            _apply_mixer(state, n, betas[layer])
    return state


def qaoa(
    qubo: QUBO,
    p: int = 2,
    restarts: int = 6,
    samples: int = 32,
    seed: int = 0,
    max_n: int = QAOA_MAX_N,
    maxiter: int = 400,
    mixer: str = "x",
    k: int | None = None,
) -> SolverResult:
    """QAOA on an exact statevector simulation.

    `mixer="x"` is standard QAOA: start from the uniform superposition, mix with
    a transverse field, and rely on a penalty term in the QUBO to discourage
    infeasible bitstrings. `mixer="xy"` starts from a Dicke state and mixes with
    a ring XY coupling, which conserves Hamming weight, so "pick exactly k"
    holds by construction and the QUBO needs no penalty at all. Pass the
    penalty-free objective QUBO with the xy mixer and the penalised one with x.

    The cost diagonal is divided by its own spread before entering the phase
    separation. Without that the useful gamma range scales like 1/|diag|, and a
    search over [0, 2*pi] wraps the phases around many times and finds nothing.
    This is a reparameterisation of gamma, not a change to the problem.

    Reports the best energy among the `samples` most probable basis states,
    which is how QAOA is used in practice: run the circuit, take shots, keep the
    best measured solution. The raw expectation value is reported alongside it,
    because the gap between the two says how much of the answer came from the
    optimisation and how much from sampling a big enough set.
    """
    if qubo.n > max_n:
        raise ValueError(f"qaoa refuses n={qubo.n} (statevector limit {max_n})")
    if mixer not in ("x", "xy"):
        raise ValueError(f"unknown mixer {mixer!r}")
    if mixer == "xy" and k is None:
        raise ValueError("the xy mixer needs k")

    t0 = time.perf_counter()
    from scipy.optimize import minimize

    n = qubo.n
    diag = cost_diagonal(qubo)
    phase_scale = float(diag.max() - diag.min()) or 1.0
    rng = np.random.default_rng(seed)

    def expectation(params: np.ndarray) -> float:
        state = qaoa_state(params, diag, n, mixer=mixer, k=k, phase_scale=phase_scale)
        return float((np.abs(state) ** 2) @ diag)

    # One deterministic linear-ramp start (gamma rising, beta falling) plus
    # random restarts. The ramp is a standard warm start for QAOA schedules.
    ramp = np.concatenate(
        [
            np.linspace(0.3, 1.5, p) * np.pi,
            np.linspace(0.6, 0.1, p) * np.pi / 2,
        ]
    )
    starts = [ramp] + [
        np.concatenate([rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi, p)])
        for _ in range(restarts)
    ]

    best = None
    for x0 in starts:
        res = minimize(expectation, x0, method="COBYLA", options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res

    state = qaoa_state(best.x, diag, n, mixer=mixer, k=k, phase_scale=phase_scale)
    probs = np.abs(state) ** 2
    top = np.argsort(probs)[::-1][:samples]
    pick = top[int(np.argmin(diag[top]))]
    x = np.array([(int(pick) >> i) & 1 for i in range(n)], dtype=np.int64)

    argmax_state = int(np.argmax(probs))
    feasible_mass = (
        float(probs[_popcount(np.arange(1 << n, dtype=np.int64)) == k].sum())
        if k is not None
        else float("nan")
    )
    return SolverResult(
        f"qaoa_{mixer}",
        x,
        float(diag[pick]),
        time.perf_counter() - t0,
        {
            "p": p,
            "mixer": mixer,
            "expectation": float(best.fun),
            "params": best.x.tolist(),
            "samples": samples,
            "top_probability": float(probs[argmax_state]),
            "argmax_energy": float(diag[argmax_state]),
            "restarts": len(starts),
            "phase_scale": phase_scale,
            "feasible_mass": feasible_mass,
        },
    )
