"""Tests for the scoring model, the QUBO encoding, and the three solvers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qcrispr.design import GuideCandidate
from qcrispr.offtarget import (
    GenomeIndex,
    LocusRegistry,
    OffTarget,
    burden,
    locus_keys,
)
from qcrispr.qubo import build_guide_qubo
from qcrispr.data import OffTargetDataset, parse_alignment
from qcrispr.evaluate import average_precision, guide_folds, spearman
from qcrispr.features import (
    BULGE_FEATURE_NAMES,
    FEATURE_NAMES,
    alignment_feature_names,
    categorical_mask,
    cfd_column,
    feature_names,
    featurize,
    features_from_alignments,
)
from qcrispr.scoring import (
    BULGE_PENALTY,
    MismatchModel,
    cfd_score,
    cfd_score_alignment,
    load_cfd_model,
    on_target_proxy,
    pam_factor,
)
from qcrispr.sequence import (
    GUIDE_LEN,
    Site,
    encode,
    find_sites,
    make_demo_genome,
    pam_matches,
    revcomp,
)
from qcrispr.solvers import (
    _apply_mixer,
    _apply_xy_ring,
    _popcount,
    brute_force,
    constrained_annealing,
    cost_diagonal,
    dicke_state,
    greedy,
    qaoa,
    qaoa_state,
    repair_to_cardinality,
    simulated_annealing,
)

# A 20nt spacer that is NOT its own reverse complement, so strand assertions
# cannot pass by accident. ACGT-repeat spacers are palindromic under revcomp.
SPACER = "ACGGTTACAGCTTGACCATA"
PALINDROME_FREE = SPACER != revcomp(SPACER)


# --------------------------------------------------------------------------
# sequence


def test_test_spacer_is_not_self_complementary():
    assert PALINDROME_FREE


def test_revcomp_roundtrip():
    seq = "ACGTTGCANNACGT"
    assert revcomp(revcomp(seq)) == seq
    assert revcomp("AACCGGTT") == "AACCGGTT"[::-1].translate(
        str.maketrans("ACGT", "TGCA")
    )


def test_encode_marks_non_acgt_as_unmatchable():
    codes = encode("ACGTN")
    assert list(codes[:4]) == [0, 1, 2, 3]
    assert codes[4] == 4
    assert not np.any(codes[4] == codes[:4])


def test_pam_matches_iupac():
    assert pam_matches("AGG", "NGG")
    assert pam_matches("TGG", "NGG")
    assert not pam_matches("AGT", "NGG")
    assert pam_matches("AAG", "NAG")


def test_find_sites_forward_coordinates():
    seq = "TTTT" + SPACER + "CGG" + "TTTT"
    sites = find_sites(seq, "c", ("NGG",), both_strands=False)
    assert len(sites) == 1
    site = sites[0]
    assert site.start == 4
    assert site.spacer == SPACER
    assert site.pam == "CGG"
    assert site.strand == "+"


def test_minus_strand_site_maps_back_to_forward_coordinates():
    # Write the reverse complement so the protospacer lives on the minus strand.
    block = revcomp(SPACER + "CGG")
    seq = "TTTTTT" + block + "TTTTTT"
    sites = find_sites(seq, "c", ("NGG",))
    minus = [s for s in sites if s.strand == "-" and s.spacer == SPACER]
    assert len(minus) == 1
    site = minus[0]
    # The spacer occupies forward [start, start+20); its PAM sits to the left.
    assert seq[site.start : site.start + GUIDE_LEN] == revcomp(SPACER)
    assert revcomp(seq[site.start - 3 : site.start]) == "CGG"


def test_site_cut_is_pam_proximal_on_both_strands():
    plus = Site("c", 100, "+", SPACER, "AGG")
    minus = Site("c", 100, "-", SPACER, "AGG")
    assert plus.cut == 117
    assert minus.cut == 103


def test_demo_genome_is_deterministic_and_planted_sites_are_findable():
    seq_a, planted_a = make_demo_genome(length=40_000, seed=3, target_start=20_000)
    seq_b, planted_b = make_demo_genome(length=40_000, seed=3, target_start=20_000)
    assert seq_a == seq_b
    assert planted_a == planted_b
    assert planted_a, "expected planted off-targets"

    # Every planted site must sit in the genome exactly where it was recorded.
    # Protospacer length varies: 21nt for a DNA bulge, 19nt for an RNA bulge.
    for row in planted_a:
        start = row["start"]
        length = len(row["protospacer"])
        assert seq_a[start : start + length] == row["protospacer"]
        assert seq_a[start + length : start + length + 3] == row["pam"]
        assert row["site_start"] == start + length - GUIDE_LEN

    # And the enumerator must find them from the guide sequence alone.
    index = GenomeIndex({"demo_chr": seq_a})
    guide = planted_a[0]["guide"]
    found = {o.site.start for o in index.find_offtargets(guide, max_mismatches=4)}
    expected = {
        r["start"] for r in planted_a
        if r["guide"] == guide and r["mismatches"] <= 4
    }
    assert expected <= found


# --------------------------------------------------------------------------
# scoring


def test_perfect_match_with_canonical_pam_scores_one():
    assert cfd_score(SPACER, SPACER, "AGG") == pytest.approx(1.0)


def test_pam_proximal_mismatch_costs_more_than_distal():
    distal = ("C" if SPACER[0] != "C" else "A") + SPACER[1:]
    proximal = SPACER[:-1] + ("C" if SPACER[-1] != "C" else "A")
    assert cfd_score(SPACER, proximal, "AGG") < cfd_score(SPACER, distal, "AGG")


def test_more_mismatches_never_increase_the_score():
    one = ("C" if SPACER[0] != "C" else "A") + SPACER[1:]
    two = ("C" if SPACER[0] != "C" else "A") + ("C" if SPACER[1] != "C" else "A") + SPACER[2:]
    assert cfd_score(SPACER, two, "AGG") <= cfd_score(SPACER, one, "AGG")


def test_pam_preference_ordering():
    assert pam_factor("AGG") > pam_factor("AAG") > pam_factor("AGA") > pam_factor("ACG")
    assert pam_factor("TTT") < pam_factor("AGT")


def test_cfd_rejects_length_mismatch():
    with pytest.raises(ValueError):
        cfd_score("ACGT", "ACG", "AGG")


def test_on_target_proxy_bounds_and_polyt_penalty():
    balanced = "ACGTACGTACGTACGTACGG"
    polyt = "ACGTTTTACGTACGTACGTG"
    for s in (balanced, polyt):
        assert 0.0 <= on_target_proxy(s) <= 1.0
    assert on_target_proxy(polyt) < on_target_proxy(balanced)


# --------------------------------------------------------------------------
# off-target enumeration


def _offtarget(start: int, cfd: float, mismatches: int = 1, chrom: str = "c"):
    return OffTarget(Site(chrom, start, "+", SPACER, "AGG"), mismatches, cfd)


def _registry(offtarget_lists, window: int = 20) -> LocusRegistry:
    return LocusRegistry.from_cuts(
        (o.locus for offs in offtarget_lists for o in offs), window=window
    )


def test_burden_sums_cfd():
    offs = [_offtarget(100, 0.4), _offtarget(104, 0.1)]
    assert burden(offs) == pytest.approx(0.5)


def test_nearby_cuts_cluster_into_one_locus():
    # Cuts at 117 and 121 are 4bp apart. Note that integer-division bucketing
    # would split them (117//20 = 5, 121//20 = 6); distance clustering must not.
    offs = [_offtarget(100, 0.4), _offtarget(104, 0.1)]
    assert [o.site.cut for o in offs] == [117, 121]
    assert len(locus_keys(offs, _registry([offs], window=20))) == 1
    assert len(locus_keys(offs, _registry([offs], window=2))) == 2


def test_locus_ids_agree_across_guides():
    # The collision the QUBO's quadratic term exists to catch: two different
    # guides cutting essentially the same place must share a locus id.
    guide_a = [_offtarget(100, 0.3)]   # cut 117
    guide_b = [_offtarget(104, 0.2)]   # cut 121
    registry = _registry([guide_a, guide_b], window=20)
    loci_a = locus_keys(guide_a, registry)
    loci_b = locus_keys(guide_b, registry)
    assert loci_a == loci_b
    assert len(loci_a & loci_b) == 1


def test_distant_cuts_stay_separate_loci():
    guide_a = [_offtarget(100, 0.3)]
    guide_b = [_offtarget(9000, 0.2)]
    registry = _registry([guide_a, guide_b], window=20)
    assert not (locus_keys(guide_a, registry) & locus_keys(guide_b, registry))


def test_single_linkage_chains_through_intermediate_cuts():
    # 117 -- 131 -- 145: each step is within 20bp, so all three are one locus.
    offs = [_offtarget(100, 0.3), _offtarget(114, 0.3), _offtarget(128, 0.3)]
    assert [o.site.cut for o in offs] == [117, 131, 145]
    assert len(locus_keys(offs, _registry([offs], window=20))) == 1


def test_unregistered_cut_becomes_its_own_locus():
    registry = _registry([[_offtarget(100, 0.3)]], window=20)
    stranger = [_offtarget(500, 0.3, chrom="other")]
    assert locus_keys(stranger, registry) == frozenset({("other", 517)})


def test_find_offtargets_excludes_the_on_target_site():
    seq = "TTTT" + SPACER + "CGG" + "TTTT"
    index = GenomeIndex({"c": seq})
    site = find_sites(seq, "c", ("NGG",), both_strands=False)[0]
    with_self = index.find_offtargets(SPACER, max_mismatches=0)
    without = index.find_offtargets(SPACER, max_mismatches=0, exclude={site.key})
    assert len(with_self) == len(without) + 1


def test_find_offtargets_rejects_wrong_length_spacer():
    index = GenomeIndex({"c": "ACGT" * 20})
    with pytest.raises(ValueError):
        index.find_offtargets("ACGT")


# --------------------------------------------------------------------------
# bulge-aware alignment


def _bulge_fixture(protospacer: str) -> tuple[str, int]:
    """Embed a protospacer plus PAM, returning the sequence and site start."""
    seq = "TTTTTT" + protospacer + "CGG" + "TTTTTT"
    pam_start = 6 + len(protospacer)
    return seq, pam_start - GUIDE_LEN


def test_dna_bulge_is_found_only_when_bulges_are_enabled():
    # An extra base in the genome: 21nt of DNA against a 20nt guide.
    protospacer = SPACER[:10] + "A" + SPACER[10:]
    assert len(protospacer) == GUIDE_LEN + 1
    seq, site_start = _bulge_fixture(protospacer)

    with_bulges = GenomeIndex({"c": seq}, max_bulges=1).find_offtargets(
        SPACER, max_mismatches=2
    )
    found = [o for o in with_bulges if o.site.start == site_start]
    assert found, "bulge-aware search must find the site"
    assert found[0].bulges == 1
    assert found[0].mismatches == 0
    assert "DNA bulge" in found[0].alignment

    mismatch_only = GenomeIndex({"c": seq}, max_bulges=0).find_offtargets(
        SPACER, max_mismatches=2
    )
    assert not [o for o in mismatch_only if o.site.start == site_start], (
        "a mismatch-only search cannot represent this site at any threshold"
    )


def test_rna_bulge_is_found_only_when_bulges_are_enabled():
    # A missing base in the genome: 19nt of DNA against a 20nt guide.
    protospacer = SPACER[:10] + SPACER[11:]
    assert len(protospacer) == GUIDE_LEN - 1
    seq, site_start = _bulge_fixture(protospacer)

    with_bulges = GenomeIndex({"c": seq}, max_bulges=1).find_offtargets(
        SPACER, max_mismatches=2
    )
    found = [o for o in with_bulges if o.site.start == site_start]
    assert found
    assert found[0].bulges == 1
    assert found[0].mismatches == 0
    assert "RNA bulge" in found[0].alignment

    mismatch_only = GenomeIndex({"c": seq}, max_bulges=0).find_offtargets(
        SPACER, max_mismatches=2
    )
    assert not [o for o in mismatch_only if o.site.start == site_start]


def test_bulge_score_is_the_pam_factor_times_the_bulge_penalty():
    # A zero-mismatch bulge alignment has no mismatch terms, so its score is
    # exactly the PAM factor times the (unvalidated) bulge penalty. This pins
    # down that bulge sites are penalised rather than scored as perfect.
    protospacer = SPACER[:10] + "A" + SPACER[10:]
    seq, site_start = _bulge_fixture(protospacer)
    hits = GenomeIndex({"c": seq}, max_bulges=1).find_offtargets(
        SPACER, max_mismatches=0
    )
    found = [o for o in hits if o.site.start == site_start][0]
    assert found.cfd == pytest.approx(pam_factor("CGG") * BULGE_PENALTY)
    assert found.cfd < pam_factor("CGG")


def test_exact_alignment_is_preferred_over_a_bulge_reading():
    # A perfect 20nt match must be reported as exact even though the index
    # also holds 19nt and 21nt windows that could be force-fit with a bulge.
    seq, site_start = _bulge_fixture(SPACER)
    hits = GenomeIndex({"c": seq}, max_bulges=1).find_offtargets(
        SPACER, max_mismatches=4
    )
    found = [o for o in hits if o.site.start == site_start][0]
    assert found.bulges == 0
    assert found.mismatches == 0
    assert found.alignment == "exact"
    assert found.cfd == pytest.approx(1.0)


def test_bulge_budget_zero_leaves_only_mismatch_windows():
    seq, _ = _bulge_fixture(SPACER)
    index = GenomeIndex({"c": seq}, max_bulges=0)
    assert index.lengths == (GUIDE_LEN,)
    index_b = GenomeIndex({"c": seq}, max_bulges=1)
    assert index_b.lengths == (GUIDE_LEN - 1, GUIDE_LEN, GUIDE_LEN + 1)


def test_demo_genome_plants_bulges_and_all_are_recoverable():
    seq, planted = make_demo_genome(length=120_000, seed=7, target_start=50_000)
    bulged = [r for r in planted if r["bulge"] != "none"]
    assert bulged, "the demo genome should plant bulge sites"
    assert {r["bulge"] for r in bulged} == {"dna", "rna"}

    index = GenomeIndex({"demo_chr": seq}, max_bulges=1)
    by_guide: dict[str, set[int]] = {}
    for row in bulged:
        guide = row["guide"]
        if guide not in by_guide:
            by_guide[guide] = {
                o.site.start
                for o in index.find_offtargets(guide, max_mismatches=4, min_cfd=0.0)
            }
        assert row["site_start"] in by_guide[guide], (
            f"missed planted {row['bulge']} bulge at {row['site_start']}"
        )


# --------------------------------------------------------------------------
# QUBO


def _candidate(start: int, burden_value: float, eff: float, loci) -> GuideCandidate:
    """A GuideCandidate with hand-set numbers, for testing the QUBO directly."""
    return GuideCandidate(
        site=Site("c", start, "+", SPACER, "AGG"),
        efficiency=eff,
        offtargets=[],
        burden=burden_value,
        loci=frozenset(loci),
    )


def test_energy_matches_the_matrix_form_including_offset():
    cands = [
        _candidate(0, 1.0, 0.5, [("c", 1)]),
        _candidate(1000, 0.5, 0.2, [("c", 1)]),
    ]
    qubo = build_guide_qubo(cands, k=1, w_shared=1.0, w_spacing=0.0, penalty=10.0)
    assert qubo.offset != 0.0
    for x in ([0, 0], [1, 0], [0, 1], [1, 1]):
        xv = np.array(x, dtype=float)
        assert qubo.energy(x) == pytest.approx(float(xv @ qubo.Q @ xv + qubo.offset))


def test_optimum_respects_the_cardinality_constraint():
    rng = np.random.default_rng(0)
    for _ in range(5):
        cands = [
            _candidate(i * 1000, float(rng.random()), float(rng.random()),
                       [("c", int(rng.integers(0, 4)))])
            for i in range(10)
        ]
        for k in (1, 3, 5):
            qubo = build_guide_qubo(cands, k=k)
            assert qubo.cardinality(brute_force(qubo).x) == k


def test_shared_loci_create_real_off_diagonal_couplings():
    cands = [
        _candidate(0, 0.5, 0.5, [("c", 7)]),
        _candidate(1000, 0.5, 0.5, [("c", 7)]),
        _candidate(2000, 0.5, 0.5, [("c", 99)]),
    ]
    qubo = build_guide_qubo(cands, k=2, w_shared=1.0, w_spacing=0.0, penalty=0.0)
    # With the cardinality penalty switched off, only the colliding pair couples.
    assert qubo.Q[0, 1] > 0
    assert qubo.Q[0, 2] == pytest.approx(0.0)


def test_pair_penalty_can_override_the_greedy_choice():
    # Guides 0 and 1 have the lowest burden but collide at one locus; guides 2
    # and 3 are worse on their own and collide with nobody. This is the case
    # that makes the problem combinatorial rather than a sort.
    cands = [
        _candidate(0, 0.00, 1.0, [("c", 5)]),
        _candidate(1000, 0.00, 1.0, [("c", 5)]),
        _candidate(2000, 0.10, 1.0, [("c", 50)]),
        _candidate(3000, 0.10, 1.0, [("c", 60)]),
    ]
    greedy = build_guide_qubo(cands, k=2, w_shared=0.0, w_spacing=0.0)
    coupled = build_guide_qubo(cands, k=2, w_shared=5.0, w_spacing=0.0)
    assert sorted(brute_force(greedy).support) == [0, 1]
    assert sorted(brute_force(coupled).support) != [0, 1]


def test_spacing_penalty_separates_nearby_cuts():
    # Two guides 10bp apart, two far away. With a spacing penalty the close
    # pair should not be chosen together.
    cands = [
        _candidate(0, 0.0, 1.0, []),
        _candidate(10, 0.0, 1.0, []),
        _candidate(5000, 0.05, 1.0, []),
        _candidate(9000, 0.05, 1.0, []),
    ]
    spaced = build_guide_qubo(cands, k=2, w_shared=0.0, w_spacing=5.0, min_spacing=50)
    assert sorted(brute_force(spaced).support) != [0, 1]


def test_build_guide_qubo_validates_k():
    cands = [_candidate(0, 0.5, 0.5, [])]
    with pytest.raises(ValueError):
        build_guide_qubo(cands, k=0)
    with pytest.raises(ValueError):
        build_guide_qubo(cands, k=2)
    with pytest.raises(ValueError):
        build_guide_qubo([], k=1)


# --------------------------------------------------------------------------
# solvers


def _random_candidates(n: int, seed: int):
    rng = np.random.default_rng(seed)
    return [
        _candidate(i * 1000, float(rng.random()), float(rng.random()),
                   [("c", int(rng.integers(0, 5)))])
        for i in range(n)
    ]


def _random_qubo(n: int, k: int, seed: int, penalty: float | None = None):
    return build_guide_qubo(_random_candidates(n, seed), k=k, penalty=penalty)


def test_cost_diagonal_indexing_matches_energy():
    qubo = _random_qubo(6, 2, seed=1)
    diag = cost_diagonal(qubo)
    rng = np.random.default_rng(2)
    for _ in range(20):
        x = rng.integers(0, 2, size=qubo.n)
        index = int(sum(int(b) << i for i, b in enumerate(x)))
        assert diag[index] == pytest.approx(qubo.energy(x))


def test_mixer_is_an_rx_rotation():
    # RX(pi) on |0> moves all amplitude to |1>.
    state = np.array([1.0 + 0j, 0.0 + 0j])
    _apply_mixer(state, 1, np.pi / 2)
    assert abs(state[0]) == pytest.approx(0.0, abs=1e-12)
    assert abs(state[1]) == pytest.approx(1.0)


def test_mixer_preserves_norm_on_many_qubits():
    rng = np.random.default_rng(7)
    n = 5
    state = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    state /= np.linalg.norm(state)
    _apply_mixer(state, n, 0.37)
    assert float(np.linalg.norm(state)) == pytest.approx(1.0)


def test_qaoa_state_stays_normalised():
    qubo = _random_qubo(6, 2, seed=3)
    diag = cost_diagonal(qubo)
    rng = np.random.default_rng(4)
    for p in (1, 2, 3):
        params = rng.uniform(0, np.pi, 2 * p)
        state = qaoa_state(params, diag, qubo.n)
        assert float(np.sum(np.abs(state) ** 2)) == pytest.approx(1.0)


def test_zero_depth_qaoa_is_the_uniform_superposition():
    qubo = _random_qubo(5, 2, seed=5)
    diag = cost_diagonal(qubo)
    probs = np.abs(qaoa_state(np.array([]), diag, qubo.n)) ** 2
    assert probs == pytest.approx(np.full(1 << qubo.n, 1.0 / (1 << qubo.n)))


def test_no_solver_beats_the_exact_optimum():
    for seed in range(3):
        qubo = _random_qubo(8, 3, seed=seed)
        optimum = brute_force(qubo).energy
        sa = simulated_annealing(qubo, restarts=5, sweeps=100, seed=seed)
        q = qaoa(qubo, p=2, restarts=2, seed=seed)
        assert sa.energy >= optimum - 1e-9
        assert q.energy >= optimum - 1e-9


def test_brute_force_and_qaoa_refuse_oversized_problems():
    qubo = _random_qubo(4, 2, seed=0)
    with pytest.raises(ValueError):
        brute_force(qubo, max_n=3)
    with pytest.raises(ValueError):
        qaoa(qubo, max_n=3)


# --------------------------------------------------------------------------
# the constrained (XY mixer / Dicke state) route


def test_popcount_matches_python():
    values = np.arange(256, dtype=np.int64)
    assert list(_popcount(values)) == [bin(v).count("1") for v in values]


def test_dicke_state_is_uniform_over_weight_k_strings():
    from math import comb

    n, k = 6, 3
    state = dicke_state(n, k)
    weights = _popcount(np.arange(1 << n, dtype=np.int64))
    probs = np.abs(state) ** 2
    assert float(probs.sum()) == pytest.approx(1.0)
    assert int(np.count_nonzero(probs)) == comb(n, k)
    assert float(probs[weights != k].sum()) == pytest.approx(0.0, abs=1e-15)
    assert probs[weights == k] == pytest.approx(1.0 / comb(n, k))


def test_dicke_state_rejects_impossible_weight():
    with pytest.raises(ValueError):
        dicke_state(3, 4)


def test_xy_mixer_conserves_hamming_weight():
    # The property the whole constrained route rests on: a Dicke start can
    # never leak probability outside the exactly-k subspace, at any beta.
    n, k = 5, 2
    state = dicke_state(n, k)
    weights = _popcount(np.arange(1 << n, dtype=np.int64))
    for beta in (0.3, 1.1, 2.7, -0.8):
        _apply_xy_ring(state, n, beta)
        probs = np.abs(state) ** 2
        assert float(probs.sum()) == pytest.approx(1.0)
        assert float(probs[weights != k].sum()) == pytest.approx(0.0, abs=1e-12)


def test_xy_mixer_actually_moves_amplitude():
    # Conserving weight is worthless if the mixer is secretly the identity.
    n, k = 4, 2
    state = dicke_state(n, k)
    before = state.copy()
    _apply_xy_ring(state, n, 0.7)
    assert not np.allclose(state, before)


def test_qaoa_xy_stays_inside_the_feasible_subspace():
    k = 3
    objective = _random_qubo(8, k, seed=0, penalty=0.0)
    result = qaoa(objective, p=2, restarts=1, seed=0, mixer="xy", k=k)
    assert int(result.x.sum()) == k
    assert result.extra["feasible_mass"] == pytest.approx(1.0)
    assert result.name == "qaoa_xy"


def test_qaoa_xy_never_beats_the_constrained_optimum():
    k = 3
    for seed in range(3):
        cands = _random_candidates(8, seed)
        penalised = build_guide_qubo(cands, k=k)
        objective = build_guide_qubo(cands, k=k, penalty=0.0)
        optimum = brute_force(penalised).energy
        result = qaoa(objective, p=2, restarts=1, seed=seed, mixer="xy", k=k)
        assert result.energy >= optimum - 1e-9


def test_penalised_and_objective_energies_agree_on_feasible_points():
    # The two encodings must coincide wherever the constraint holds, otherwise
    # comparing the x-mixer and xy-mixer energies is meaningless.
    k = 3
    cands = _random_candidates(7, seed=11)
    penalised = build_guide_qubo(cands, k=k)
    objective = build_guide_qubo(cands, k=k, penalty=0.0)

    rng = np.random.default_rng(12)
    for _ in range(10):
        x = np.zeros(len(cands), dtype=np.int64)
        x[rng.choice(len(cands), size=k, replace=False)] = 1
        assert penalised.energy(x) == pytest.approx(objective.energy(x))

    infeasible = np.zeros(len(cands), dtype=np.int64)
    infeasible[:k - 1] = 1
    assert penalised.energy(infeasible) > objective.energy(infeasible)


def test_qaoa_validates_mixer_arguments():
    qubo = _random_qubo(5, 2, seed=0)
    with pytest.raises(ValueError):
        qaoa(qubo, mixer="nonsense")
    with pytest.raises(ValueError):
        qaoa(qubo, mixer="xy")  # no k
    with pytest.raises(ValueError):
        qaoa_state(np.array([0.1, 0.1]), cost_diagonal(qubo), qubo.n, mixer="xy")


# --------------------------------------------------------------------------
# the published CFD tables


DATA = Path(__file__).resolve().parent.parent / "data"
CFD_MISMATCH = DATA / "cfd_mismatch.tsv"
CFD_PAM = DATA / "cfd_pam.tsv"
needs_cfd = pytest.mark.skipif(
    not (CFD_MISMATCH.exists() and CFD_PAM.exists()),
    reason="run scripts/import_cfd.py first",
)


@needs_cfd
def test_published_table_loads_with_the_canonical_shape():
    model = load_cfd_model(CFD_MISMATCH, CFD_PAM)
    assert model.is_published
    assert len(model.table) == 12 * 20
    assert len(model.pam) == 16


@needs_cfd
def test_published_weights_match_known_published_values():
    # Spot checks against values verified in the source tables. The key detail
    # is the convention: a CFD key 'rA:dG,1' names the complement of the base
    # seen in the protospacer, so it is stored here as guide A vs observed C.
    model = load_cfd_model(CFD_MISMATCH, CFD_PAM)
    assert model.retention(1, "A", "C") == pytest.approx(0.857142857)
    assert model.pam_factor("AGG") == pytest.approx(1.0)
    assert model.pam_factor("CAG") == pytest.approx(0.259259259)
    assert model.pam_factor("TGA") == pytest.approx(0.069444444)


@needs_cfd
def test_pam_factor_ignores_the_first_base():
    model = load_cfd_model(CFD_MISMATCH, CFD_PAM)
    assert model.pam_factor("AGG") == model.pam_factor("TGG")
    assert model.pam_factor("AAA") == pytest.approx(0.0)


@needs_cfd
def test_published_model_refuses_to_fall_back_silently():
    # A missing cell must raise, not quietly return a parameterised guess:
    # mixing the two models would make scores untraceable.
    model = load_cfd_model(CFD_MISMATCH, CFD_PAM)
    with pytest.raises(KeyError):
        model.retention(21, "A", "C")


@needs_cfd
def test_published_model_scores_a_perfect_match_as_one():
    model = load_cfd_model(CFD_MISMATCH, CFD_PAM)
    assert cfd_score(SPACER, SPACER, "AGG", model=model) == pytest.approx(1.0)


@needs_cfd
def test_parameterisation_is_measurably_worse_than_published():
    # Documents the gap rather than asserting the approximation is fine.
    published = load_cfd_model(CFD_MISMATCH, CFD_PAM)
    approx_model = MismatchModel()
    errors = [
        abs(approx_model.retention(pos, g, d) - value)
        for (pos, g, d), value in published.table.items()
    ]
    mean_error = sum(errors) / len(errors)
    assert mean_error > 0.1, (
        "if the parameterisation got this close to published CFD, re-check "
        "that the published table is actually being loaded"
    )


# --------------------------------------------------------------------------
# metrics and features for the measured-data evaluation


def test_average_precision_matches_sklearn():
    # The whole evaluation rests on this metric, so it is checked against the
    # reference implementation rather than trusted.
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    for _ in range(10):
        n = int(rng.integers(50, 400))
        labels = (rng.random(n) < 0.05).astype(np.int8)
        if labels.sum() == 0:
            labels[0] = 1
        scores = rng.random(n)
        assert average_precision(labels, scores) == pytest.approx(
            average_precision_score(labels, scores)
        )


def test_average_precision_is_one_for_a_perfect_ranking():
    labels = np.array([1, 1, 0, 0, 0], dtype=np.int8)
    assert average_precision(labels, np.array([9.0, 8.0, 1.0, 0.5, 0.0])) == pytest.approx(1.0)


def test_average_precision_undefined_without_both_classes():
    assert np.isnan(average_precision(np.zeros(5, np.int8), np.arange(5.0)))
    assert np.isnan(average_precision(np.ones(5, np.int8), np.arange(5.0)))


def test_spearman_matches_scipy_including_ties():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(1)
    for _ in range(10):
        n = int(rng.integers(20, 200))
        x = rng.random(n)
        # Heavy ties on y, which is the realistic case: most read counts are 0.
        y = np.where(rng.random(n) < 0.8, 0.0, rng.random(n) * 100)
        assert spearman(x, y) == pytest.approx(float(spearmanr(x, y).statistic))


def test_spearman_is_one_for_a_monotone_relationship():
    x = np.arange(10.0)
    assert spearman(x, 3 * x + 1) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_guide_folds_partition_every_guide_exactly_once():
    folds = guide_folds(110, 10, seed=0)
    assert len(folds) == 10
    combined = np.concatenate(folds)
    assert sorted(combined.tolist()) == list(range(110))
    for a_index, a in enumerate(folds):
        for b in folds[a_index + 1 :]:
            assert not set(a.tolist()) & set(b.tolist())


def _tiny_dataset(reads) -> OffTargetDataset:
    guide = SPACER + "AGG"
    protos = [guide, "C" + guide[1:], guide[:19] + "A" + guide[20:]]
    return OffTargetDataset(
        guides=np.array([guide]),
        guide_idx=np.zeros(len(protos), dtype=np.int16),
        proto=np.vstack([encode(p) for p in protos]),
        reads=np.asarray(reads, dtype=np.float32),
        distance=np.array([0, 1, 1], dtype=np.uint8),
    )


def test_dataset_labels_respect_the_threshold():
    dataset = _tiny_dataset([500.0, 50.0, 0.0])
    assert list(dataset.labels(100.0)) == [1, 0, 0]
    assert list(dataset.labels(10.0)) == [1, 1, 0]


@needs_cfd
def test_featurised_cfd_column_matches_the_scorer():
    # The cfd feature must equal what scoring.py computes for the same
    # alignment, or the baseline and the model see different quantities.
    model = load_cfd_model(CFD_MISMATCH, CFD_PAM)
    dataset = _tiny_dataset([500.0, 50.0, 0.0])
    matrix = featurize(dataset, model)
    assert matrix.shape == (3, len(feature_names(extended=True)))
    assert featurize(dataset, model, extended=False).shape == (3, len(FEATURE_NAMES))
    # The cfd column must sit at the same index in both variants, since the
    # extended columns are appended rather than interleaved.
    assert np.allclose(
        cfd_column(matrix), cfd_column(featurize(dataset, model, extended=False))
    )

    guide = str(dataset.guides[0])
    for row in range(len(dataset)):
        protospacer = "".join("ACGTN"[c] for c in dataset.proto[row])
        triples = [
            (i + 1, g, d)
            for i, (g, d) in enumerate(zip(guide[:20], protospacer[:20]))
            if g != d
        ]
        expected = cfd_score_alignment(triples, protospacer[20:], model=model)
        assert cfd_column(matrix)[row] == pytest.approx(expected, rel=1e-5)


@needs_cfd
def test_perfect_match_features_are_self_consistent():
    model = load_cfd_model(CFD_MISMATCH, CFD_PAM)
    dataset = _tiny_dataset([500.0, 50.0, 0.0])
    matrix = featurize(dataset, model)
    names = {name: i for i, name in enumerate(FEATURE_NAMES)}
    # Row 0 is the on-target: no mismatches, canonical PAM, CFD 1.0.
    assert matrix[0, names["mismatch_count"]] == 0
    assert matrix[0, names["cfd"]] == pytest.approx(1.0)
    assert matrix[0, names["retention_product"]] == pytest.approx(1.0)
    assert matrix[0, names["retention_deficit"]] == pytest.approx(0.0)
    # Row 1 has one mismatch at position 1, far from the PAM.
    assert matrix[1, names["mismatch_count"]] == 1
    assert matrix[1, names["most_pam_distal_mm"]] == 1
    assert matrix[1, names["mm_pos_1_5"]] == 1


@needs_cfd
def test_per_position_columns_encode_mismatch_identity():
    model = load_cfd_model(CFD_MISMATCH, CFD_PAM)
    dataset = _tiny_dataset([500.0, 50.0, 0.0])
    matrix = featurize(dataset, model, extended=True)
    names = feature_names(extended=True)
    first = names.index("mm_type_pos_1")

    # Row 0 is the on-target: every position a match, so every column is 0.
    assert matrix[0, first : first + 20].max() == 0
    # Row 1 has one mismatch at position 1 only.
    assert matrix[1, first] > 0
    assert matrix[1, first + 1 : first + 20].max() == 0
    # Codes stay inside the range a categorical split can use.
    block = matrix[:, first : first + 20]
    assert block.min() >= 0 and block.max() <= 12


def test_categorical_mask_marks_only_the_position_columns():
    mask = categorical_mask(extended=True)
    names = feature_names(extended=True)
    assert mask.sum() == 20
    assert all(names[i].startswith("mm_type_pos_") for i in np.flatnonzero(mask))
    assert not categorical_mask(extended=False).any()


@needs_cfd
def test_both_feature_paths_agree_on_mismatch_only_sites():
    # There are now two featurisers: a vectorised one over fixed-width arrays
    # and a per-record one that can express bulges. On mismatch-only input
    # they must produce identical numbers, or the bulge evaluation and the
    # main evaluation are measuring different things.
    model = load_cfd_model(CFD_MISMATCH, CFD_PAM)
    dataset = _tiny_dataset([500.0, 50.0, 0.0])
    array_path = featurize(dataset, model, extended=True)

    guide = str(dataset.guides[0])
    records = []
    for row in range(len(dataset)):
        observed = "".join("ACGTN"[c] for c in dataset.proto[row])
        mismatches = [
            (i + 1, guide[i], observed[i])
            for i in range(20)
            if guide[i] != observed[i]
        ]
        records.append((guide[:20], mismatches, observed[20:], None, 0, 0))
    record_path = features_from_alignments(records, model, extended=True)

    shared = len(feature_names(extended=True))
    assert record_path.shape[1] == shared + len(BULGE_FEATURE_NAMES)
    np.testing.assert_allclose(
        record_path[:, :shared], array_path, rtol=1e-5, atol=1e-6
    )
    # With no bulge, every bulge column is zero.
    assert record_path[:, shared:].max() == 0


@needs_cfd
def test_bulge_columns_describe_the_gap():
    model = load_cfd_model(CFD_MISMATCH, CFD_PAM)
    names = alignment_feature_names(extended=False)
    records = [
        (SPACER, [], "AGG", None, 0, 0),
        (SPACER, [], "AGG", "dna", 7, 1),
        (SPACER, [], "AGG", "rna", 15, 1),
        (SPACER, [], "AGG", "both", 4, 2),
    ]
    matrix = features_from_alignments(records, model, extended=False)
    column = {name: i for i, name in enumerate(names)}

    assert matrix[0, column["n_bulges"]] == 0
    assert matrix[1, column["bulge_is_dna"]] == 1
    assert matrix[1, column["bulge_is_rna"]] == 0
    assert matrix[1, column["bulge_position"]] == 7
    assert matrix[1, column["bulge_distance_from_pam"]] == 13
    assert matrix[2, column["bulge_is_rna"]] == 1
    assert matrix[2, column["bulge_position"]] == 15
    # An alignment with gaps on both sides sets both flags and counts both.
    assert matrix[3, column["bulge_is_dna"]] == 1
    assert matrix[3, column["bulge_is_rna"]] == 1
    assert matrix[3, column["n_bulges"]] == 2

    # The cfd column must not carry a bulge penalty: all four are perfect
    # matches on their aligned positions, so all four score the same.
    for row in (1, 2, 3):
        assert matrix[0, column["cfd"]] == pytest.approx(matrix[row, column["cfd"]])


@needs_cfd
def test_parse_alignment_handles_both_gap_directions():
    # A gap in the guide means an extra DNA base; a gap in the off-target
    # means the DNA is missing one. Guide positions must be counted over the
    # ungapped spacer, since that is how CFD indexes its weights.
    guide = "G" * 20 + "NGG"

    # DNA bulge: an extra base inserted into the observed sequence.
    parsed = parse_alignment("G" * 10 + "-" + "G" * 10 + "NGG", "G" * 21 + "AGG")
    assert parsed is not None
    spacer, mismatches, pam, kind, position, n_bulges = parsed
    assert spacer == "G" * 20 and pam == "AGG"
    assert kind == "dna" and n_bulges == 1 and mismatches == []

    # RNA bulge: the observed sequence is missing a base the guide has.
    parsed = parse_alignment(guide, "G" * 10 + "-" + "G" * 9 + "AGG")
    assert parsed is not None
    spacer, mismatches, pam, kind, position, n_bulges = parsed
    assert kind == "rna" and n_bulges == 1 and position == 11
    assert mismatches == []

    # A mismatch after a gap must be numbered by ungapped guide position.
    parsed = parse_alignment(guide, "G" * 10 + "-" + "G" * 8 + "A" + "AGG")
    assert parsed is not None
    _, mismatches, _, kind, _, _ = parsed
    assert kind == "rna"
    assert mismatches == [(20, "G", "A")]


def test_parse_alignment_rejects_malformed_rows():
    assert parse_alignment("GGG", "GG") is None            # length mismatch
    assert parse_alignment("G" * 23, "G" * 20 + "A-G") is None  # gap in the PAM
    assert parse_alignment("G" * 10, "G" * 10) is None     # spacer too short


def test_featurize_refuses_the_parameterised_model():
    # Training on approximate weights and reporting it as CFD-based would be
    # untraceable, so this is a hard error rather than a warning.
    dataset = _tiny_dataset([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="published CFD"):
        featurize(dataset, MismatchModel())


# --------------------------------------------------------------------------
# the fair classical baselines


def test_greedy_selects_exactly_k():
    objective = _random_qubo(10, 3, seed=0, penalty=0.0)
    result = greedy(objective, 3)
    assert int(result.x.sum()) == 3
    assert result.energy == pytest.approx(objective.energy(result.x))


def test_greedy_is_not_always_optimal():
    # The instance family has to be able to defeat greedy, or no result about
    # any other solver on it means anything.
    #
    # Note that greedy here is strong: it re-evaluates the full quadratic
    # objective at every step, so the obvious trap (two cheap guides that
    # collide) does not catch it -- it simply declines the second one. Only a
    # linear-only ranker, like `design.prefilter`, falls for that.
    #
    # What does defeat it is a decoy. Guide 0 is the best single choice and
    # collides with everything, so taking it first forecloses the genuinely
    # best pair {2, 3}, which collides with nothing:
    #
    #   singles: 0 -> -1.0,  1 -> 0.0,  2 -> -0.6,  3 -> -0.6
    #   pairs:   {0,j} carry +1.0 collision, {2,3} carries none
    #   so {2,3} = -1.2 beats greedy's {0,2} = -0.6
    cands = [
        _candidate(0, 0.00, 1.0, [("c", 1), ("c", 2), ("c", 3)]),
        _candidate(1000, 0.05, 1.0, [("c", 1)]),
        _candidate(2000, 0.02, 1.0, [("c", 2)]),
        _candidate(3000, 0.02, 1.0, [("c", 3)]),
    ]
    objective = build_guide_qubo(cands, k=2, w_shared=1.0, w_spacing=0.0, penalty=0.0)
    penalised = build_guide_qubo(cands, k=2, w_shared=1.0, w_spacing=0.0)

    greedy_result = greedy(objective, 2)
    exact = brute_force(penalised)
    assert sorted(exact.support) == [2, 3]
    assert 0 in greedy_result.support
    assert greedy_result.energy > exact.energy + 1e-9


def test_constrained_annealing_stays_feasible_and_never_beats_the_optimum():
    k = 4
    for seed in range(3):
        cands = _random_candidates(12, seed)
        penalised = build_guide_qubo(cands, k=k)
        objective = build_guide_qubo(cands, k=k, penalty=0.0)
        optimum = brute_force(penalised).energy
        result = constrained_annealing(objective, k, restarts=5, sweeps=100, seed=seed)
        assert int(result.x.sum()) == k
        assert result.energy >= optimum - 1e-9


def test_swap_delta_matches_full_recomputation():
    # The swap neighbourhood is driven by an incremental delta; if that formula
    # is wrong the search silently explores the wrong landscape.
    objective = _random_qubo(9, 3, seed=5, penalty=0.0)
    Q = objective.Q
    rng = np.random.default_rng(6)
    x = np.zeros(objective.n)
    x[rng.choice(objective.n, size=3, replace=False)] = 1.0
    qx = Q @ x
    for i in np.flatnonzero(x):
        for j in np.flatnonzero(x == 0):
            i, j = int(i), int(j)
            delta = 2.0 * (qx[j] - qx[i]) + Q[i, i] + Q[j, j] - 2.0 * Q[i, j]
            y = x.copy()
            y[i], y[j] = 0.0, 1.0
            assert delta == pytest.approx(objective.energy(y) - objective.energy(x))


def test_repair_fixes_both_too_many_and_too_few():
    objective = _random_qubo(8, 3, seed=7, penalty=0.0)
    too_many = np.ones(8, dtype=np.int64)
    too_few = np.zeros(8, dtype=np.int64)
    assert int(repair_to_cardinality(too_many, objective, 3).sum()) == 3
    assert int(repair_to_cardinality(too_few, objective, 3).sum()) == 3


def test_repair_leaves_a_feasible_solution_alone():
    objective = _random_qubo(8, 3, seed=8, penalty=0.0)
    x = np.zeros(8, dtype=np.int64)
    x[[1, 4, 6]] = 1
    assert np.array_equal(repair_to_cardinality(x, objective, 3), x)


def test_phase_rescaling_is_recorded():
    qubo = _random_qubo(6, 2, seed=0)
    result = qaoa(qubo, p=1, restarts=1, seed=0)
    assert result.extra["phase_scale"] > 1.0
    assert result.name == "qaoa_x"
