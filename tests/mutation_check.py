"""Mutation check for the defects found by the external code audits.

For every defect: copy scripts/ and tests/ to a fresh temporary tree, apply
the mutation to one script, and run that defect's DESIGNATED regression test
in a fresh Python subprocess. The defect counts as caught only if

  * the same test PASSES on an unmodified copy of the tree (baseline), and
  * on the mutant it FAILS with an assertion or a domain error.

Outcomes are classified, never lumped together:
  pass          the test ran and passed
  skip          the test skipped itself (never counts as a detection)
  fail          AssertionError / pytest Failed, or any other non-infrastructure
                exception raised by the code under test (a detection)
  infra_error   ImportError, ModuleNotFoundError, FileNotFoundError, or a crashed
                subprocess (never counts as a detection)
Each failure records its exception type, message and traceback.

A no-op control mutation (a comment) must SURVIVE: if its designated test
fails, the harness itself is unreliable and the run is rejected.

    python tests/mutation_check.py            # exit 0 only if every check holds
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BNV, DUMP, FC = "bank_normalized_verification.py", "dump_verifier_vectors.py", "fusion_controls.py"
PRE, REP = "preflight_vectors.py", "vector_repeatability.py"
BB, BBP, T_BB = "bbnv_core.py", "bbnv_partition.py", "test_bbnv.py"
BBO, BBV, BBM, BBX = "bbnv_oracle.py", "bbnv_provenance.py", "bbnv_manifest.py", "bbnv_extract.py"
BBT, BBS, BBA = "bbnv_train.py", "bbnv_score.py", "bbnv_analyze.py"
BBF, BBPO = "bbnv_transfer.py", "bbnv_policy.py"
CT, CSW = "cirr_test.py", "cirr_split_windows.py"
T_BNV, T_DUMP, T_FC = ("test_bank_normalized_verification.py", "test_dump_verifier_vectors.py",
                       "test_fusion_controls.py")
T_RES = "test_vector_resume.py"

# (label, script, old, new, test module, designated test)
MUTANTS = [
    ("CSLS back to s - h", BNV, "CSLS_COEF = 0.5", "CSLS_COEF = 1.0",
     T_BNV, "test_csls_coefficient_is_half"),
    ("no vector finiteness check", BNV, "        _finite(v, name)\n    if not np.array_equal",
     "    if not np.array_equal", T_BNV, "test_nonfinite_vectors_are_rejected"),
    ("no vector reproduction check", BNV,
     '            raise ProtocolError("vectors do not reproduce cached %s (%.3g > %.3g)"\n'
     '                                % (term, err, VECTOR_TOL[term]))', "            pass",
     T_BNV, "test_misaligned_rows_are_rejected"),
    ("one tolerance for every BNV field", BNV, "        if err > VECTOR_TOL[term]:",
     '        if err > VECTOR_TOL["base_score"]:', T_BNV, "test_vector_tolerance_is_per_field"),
    ("tolerance policy not in provenance", BNV,
     '                   "vector_tolerance", "cache_tolerance")', "                   )",
     T_BNV, "test_decide_rejects_incompatible_results"),
    ("exclusion applied to every row", BNV, "rows = np.nonzero(ex >= 0)[0]",
     "rows = np.arange(len(ex))", T_BNV, "test_topk_means_mixed_exclusion_and_all_mean"),
    ("Holm without step-down max", BNV, "adj = max(prev, min(1.0, p * (len(order) - r)))",
     "adj = min(1.0, p * (len(order) - r))", T_BNV, "test_decide_applies_holm"),
    ("ordinal ranks for ties", BNV, "ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0",
     "ranks[order[i:j + 1]] = np.arange(i, j + 1) + 1.0",
     T_BNV, "test_spearman_handles_ties_and_constants"),
    ("constant input not undefined", BNV, '        return None, "one variable is constant"\n',
     "        pass\n", T_BNV, "test_spearman_handles_ties_and_constants"),
    ("unstable scalar sigmoid", BNV,
     "    if z >= 0.0:\n        return 1.0 / (1.0 + math.exp(-z))",
     "    if True:\n        return 1.0 / (1.0 + math.exp(-z))",
     T_BNV, "test_calibrate_is_stable_for_large_logits"),
    ("deployed parity ignores score values", BNV, "    if worst > TOL_PARITY:", "    if False:",
     T_BNV, "test_deployed_control_parity"),
    ("deployed parity ignores hit changes", BNV, "    if hit_changed:", "    if False:",
     T_BNV, "test_deployed_control_parity"),
    ("deployed parity on Hit@1 only", BNV,
     "        if any(hit_at(fc, rels[i], at) != hit_at(fr, rels[i], at) for at in (1, 5)):",
     "        if any(hit_at(fc, rels[i], at) != hit_at(fr, rels[i], at) for at in (1,)):",
     T_BNV, "test_deployed_control_parity"),
    ("checkpoints not compared", BNV,
     '    if len({m["checkpoint_sha256"] for _, _, _, m in loaded}) != 1:', "    if False:",
     T_BNV, "test_fiq_rejects_mixed_checkpoints"),
    ("missing manifest metadata accepted", BNV,
     '    if not _is_digest(man.get("config_sha256")):', "    if False:",
     T_BNV, "test_manifest_metadata_is_required"),
    ("decide ignores provenance", BNV,
     '            raise ProtocolError("results disagree on %s; they are not one experiment" % key)',
     "            pass", T_BNV, "test_decide_rejects_incompatible_results"),
    ("decide ignores CIRR source identity", BNV,
     '        raise ProtocolError("the two results use different CIRR cache/vector inputs")',
     "        pass", T_BNV, "test_decide_rejects_mismatched_or_malformed_inputs"),
    ("decide accepts any p-value", BNV, '    if not 0.0 <= t["p_two_sided"] <= 1.0:',
     "    if False:", T_BNV, "test_decide_rejects_invalid_statistics"),
    ("decide accepts reversed intervals", BNV, '    if t["ci_low"] > t["ci_high"]:',
     "    if False:", T_BNV, "test_decide_rejects_invalid_statistics"),
    ("bank subsets may touch evaluation", BNV,
     "    return {seed: sorted(random.Random(seed).sample(list(bank_idx), m))",
     "    return {seed: sorted(random.Random(seed).sample(list(range(2 * len(bank_idx))), m))",
     T_BNV, "test_bank_subsets_never_touch_the_evaluation_half"),
    ("NaN differences ignored", DUMP,
     '        if not np.all(np.isfinite(d)):\n            raise ValueError("non-finite difference in %s" % key)\n',
     "", T_DUMP, "test_reproduction_errors_zero_on_exact_and_raise_on_nan"),
    ("short fields accepted", DUMP, "                if len(r[f]) != k:\n                    raise ValueError",
     "                if False:\n                    raise ValueError",
     T_DUMP, "test_load_cache_rejects_short_field"),
    ("empty cache accepted", DUMP,
     '    if not recs:\n        raise ValueError("%s contains no records" % path)\n', "",
     T_DUMP, "test_load_cache_rejects_empty_file"),
    ("orphan manifest not detected", DUMP, "    if has_man and not has_npz:", "    if False:",
     T_DUMP, "test_output_state_covers_every_interrupted_publication"),
    ("unstable numpy sigmoid", DUMP, "    e = np.exp(-np.abs(z))", "    e = np.exp(-z)",
     T_DUMP, "test_numpy_sigmoid_is_warning_free_at_extreme_logits"),
    ("verifier terms at the base tolerance", DUMP,
     'VECTOR_TOL = {"recon_cos": 2e-4, "invariant_cos": 1e-4, "edit_cos": 1e-4,',
     'VECTOR_TOL = {"recon_cos": 1e-3, "invariant_cos": 1e-3, "edit_cos": 1e-3,',
     T_DUMP, "test_tolerance_violations_per_field"),
    ("only the fields present are checked", DUMP,
     '    if missing:\n        raise ValueError("reproduction errors missing for %s" % missing)\n'
     '    return ["%s %.3g > %.3g" % (f, worst[f], VECTOR_TOL[f])\n'
     '            for f in sorted(VECTOR_TOL) if worst[f] > VECTOR_TOL[f]]',
     '    return ["%s %.3g > %.3g" % (f, worst[f], VECTOR_TOL[f])\n'
     '            for f in sorted(worst) if worst[f] > VECTOR_TOL[f]]',
     T_DUMP, "test_tolerance_violations_per_field"),
    ("reproduction in float32", DUMP,
     "    f64 = {name: np.asarray(arr[name], dtype=np.float64)",
     "    f64 = {name: np.asarray(arr[name], dtype=np.float32)",
     T_DUMP, "test_reproduction_is_evaluated_in_float64"),
    ("skip ignores run identity", DUMP, "    if differ:\n", "    if False:\n",
     T_DUMP, "test_complete_output_requires_this_runs_identity"),
    ("skip ignores vector digest", DUMP,
     '    if man.get("schema") != SCHEMA or man.get("npz_sha256") != sha256_file(paths["npz"]):',
     '    if man.get("schema") != SCHEMA:',
     T_DUMP, "test_complete_output_requires_this_runs_identity"),
    ("probe may be skipped", DUMP,
     "    if args.skip_if_complete and args.max_queries is not None:", "    if False:",
     T_DUMP, "test_skip_if_complete_is_refused_for_a_probe"),
    ("preflight skips the analysis loader", PRE, "    load_pair(cirr_cache, cirr_npz)\n", "",
     T_RES, "test_preflight_runs_the_analysis_loader_on_cirr"),
    ("preflight ignores FashionIQ count", PRE, "        if n != FIQ_EXPECTED[cat]:",
     "        if False:", T_RES, "test_preflight_rejects_bad_fashioniq_states"),
    ("repeatability assumes no top-1 flips", REP,
     "        \"fused_top1\": sum(fused[0][i][0] != fused[1][i][0] for i in idx),",
     "        \"fused_top1\": 0,", T_RES, "test_perturbed_probe_differences_and_flips_are_counted"),
    ("repeatability accepts an unvaried pair", REP, "    if bs[0] == bs[1]:", "    if False:",
     T_RES, "test_repeatability_refuses_an_uninformative_or_mismatched_pair"),
    ("bool/fraction positions accepted", FC,
     "if tp is not None and not (_is_index(tp) and (tp == -1 or 0 <= tp < n)):",
     "if tp is not None and not (tp == -1 or 0 <= tp < n):",
     T_FC, "test_validator_rejects_malformed_positions_and_records"),
    ("null multi-positive field accepted", FC,
     '    if "relevant_positions_in_candidates" in rec:',
     '    if rec.get("relevant_positions_in_candidates") is not None:',
     T_FC, "test_validator_rejects_malformed_positions_and_records"),
    ("McNemar truncates unequal arms", FC,
     '    if len(a) != len(b):\n        raise ValueError("arms differ in length (%d vs %d)" % (len(a), len(b)))\n',
     "", T_FC, "test_mcnemar_rejects_unequal_or_nonbinary_arms"),
    # BBNV Phase A (v3.1 §6 invariants and §8 corrections)
    ("BBNV marginal ignores the prior", BB,
     "    log_marg = logsumexp(log_rho[:, None] + log_f[:, win], axis=0)",
     "    log_marg = logsumexp(log_f[:, win], axis=0)",
     T_BB, "test_inv3_induced_inverse_equals_direct_bayes_on_enumerated_joint"),
    ("BBNV undefined PMI not rejected", BB,
     "    if not np.all(np.isfinite(log_marg)):\n", "    if False:\n",
     T_BB, "test_zero_marginal_mass_makes_pmi_undefined"),
    ("BBNV energy mixture (components unnormalised)", BB,
     "    comp = log_softmax(logits, axis=-1)", "    comp = logits",
     T_BB, "test_inv1_mixture_is_normalised_and_pi_are_masses"),
    ("BBNV chunks combined by max", BB,
     "acc = part if acc is None else np.logaddexp(acc, part)",
     "acc = part if acc is None else np.maximum(acc, part)",
     T_BB, "test_inv5_chunked_marginal_equals_unchunked"),
    ("BBNV duplicate window accepted", BB, "    if np.unique(win).size != win.size:",
     "    if False:", T_BB, "test_inv6_permutation_duplicates_and_in_bank_reference"),
    ("BBNV in-bank reference added twice", BB, "            if i in pos:", "            if False:",
     T_BB, "test_inv2_query_reference_has_positive_prior_also_when_in_bank"),
    ("BBNV delta from uncentred scores", BB, '"delta": c_learned - c_induced,',
     '"delta": np.asarray(ie_learned) - np.asarray(ie_induced),',
     T_BB, "test_inv7_delta_is_exactly_the_difference_of_the_supplied_features"),
    ("BBNV heads summed, not averaged", BB, "    return np.mean(zs, axis=0) + gamma * zscore(edit)",
     "    return np.sum(zs, axis=0) + gamma * zscore(edit)",
     T_BB, "test_integrate_keeps_edit_weight_fixed_across_one_and_two_head_arms"),
    ("BBNV SD floor removed", BB, "    return np.zeros_like(c) if sd < floor else c / sd",
     "    return c / sd", T_BB, "test_zscore_floor_zeroes_only_the_degenerate_component"),
    ("BBNV AUROC ties counted as wins", BB,
     "(np.sum(diff > 0) + 0.5 * np.sum(diff == 0))", "(np.sum(diff >= 0))",
     T_BB, "test_pairwise_auroc_half_credit_for_ties_and_undefined_cases"),
    ("BBNV seed pairs not cyclic", BB,
     "SEED_PAIRS: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 2), (2, 0))",
     "SEED_PAIRS: Tuple[Tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))",
     T_BB, "test_seed_pairs_are_the_fixed_cyclic_pairs"),
    ("BBNV crossing triplet accepted", BBP, "        if p_ref != p_tgt:\n", "        if False:\n",
     T_BB, "test_split_is_role_disjoint_and_fully_accounted"),
    # linked-group partition, amendment §10 (20 Oct 2026)
    ("BBNV group key depends on order", BBP, "                parent[max(ra, rb)] = min(ra, rb)",
     "                parent[ra] = rb", T_BB, "test_linked_groups_are_components_keyed_by_their_smallest_id"),
    ("BBNV greedy ignores target shares", BBP,
     "        name = max(PARTITION_NAMES, key=lambda n: target[n] - got[n])",
     "        name = PARTITION_NAMES[0]",
     T_BB, "test_group_allocation_keeps_every_triplet_and_meets_the_shares"),
    ("BBNV groups ignore extra positives", BBP, "    group = linked_groups(positives, images)",
     "    group = linked_groups(known_positives(triplets), images)",
     T_BB, "test_group_allocation_keeps_every_triplet_and_meets_the_shares"),
    ("BBNV partial soft matches not positives", BBM, "            elif val > 0 and img != tgt:",
     "            elif val == 1.0 and img != tgt:",
     T_BB, "test_manifest_parses_cirr_rows_and_defines_positives"),
    ("BBNV unknown soft value accepted", BBM,
     "or float(val) not in SOFT_VALUES:", "or False:",
     T_BB, "test_manifest_parses_cirr_rows_and_defines_positives"),
    # group-manifest source audit (20 Oct 2026)
    ("BBNV validator skips partition agreement", BBM,
     '        need(t["partition"] == pof[t["reference"]] == pof[t["target"]], "pairid %r crosses" % (t["pairid"],))',
     '        need(True, "")', T_BB, "test_written_manifest_is_semantically_validated"),
    ("BBNV manifest published before validation", BBM,
     "    validate_written_manifest(staging, captions_obj, split_obj)\n"
     "    os.rmdir(out_dir)                                   # the empty directory reserved at start-up\n"
     "    os.replace(staging, out_dir)\n",
     "    os.rmdir(out_dir)\n    os.replace(staging, out_dir)\n"
     "    validate_written_manifest(out_dir, captions_obj, split_obj)\n",
     T_BB, "test_written_manifest_is_semantically_validated"),
    ("BBNV reference anomaly ignores its label", BBM,
     '                anomalies.append({"pairid": pid, "kind": "reference_soft_label", "label": _label(val)})',
     '                anomalies.append({"pairid": pid, "kind": "reference_soft_label", "label": "1.0"})',
     T_BB, "test_manifest_parses_cirr_rows_and_defines_positives"),
    # CIRR test submission (23 Oct 2026)
    ("CIRR submission accepts extra keys", CT, "    if keys != {str(p) for p in pairids}:",
     "    if not {str(p) for p in pairids} <= keys:", T_BB, "test_cirr_submission_format_is_enforced"),
    ("CIRR ranks 11-50 dropped", CT, "    return [top50[i] for i in window_order] + list(top50[k:])",
     "    return [top50[i] for i in window_order]", T_BB, "test_cirr_rankings_follow_the_frozen_systems"),
    ("CIRR T4 ignores whether fusion changed the top-1", CT,
     '        out["T4"].append(fused if (v["changed"] and accept4[q]) else top)',
     '        out["T4"].append(fused if accept4[q] else top)',
     T_BB, "test_cirr_rankings_follow_the_frozen_systems"),
    ("CIRR window order not compared with the cache", CSW,
     '    if any(list(map(int, c)) != r["candidate_gallery_ids"] for c, r in zip(cand, recs)):',
     '    if any(sorted(map(int, c)) != sorted(r["candidate_gallery_ids"]) for c, r in zip(cand, recs)):',
     T_BB, "test_split_windows_must_reproduce_the_cache"),
    # cross-fitted policy comparison (22 Oct 2026)
    ("BBNV policy matched acceptance ignores direction", BBPO,
     "    order = sorted(idx, key=lambda i: ((-scores[i]) if higher_accepts else scores[i], i))",
     "    order = sorted(idx, key=lambda i: (-scores[i], i))",
     T_BB, "test_policy_matched_acceptance_uses_scores_only"),
    ("BBNV policy bootstrap ignores clusters", BBPO,
     "    keys, inv = np.unique(cl, return_inverse=True)",
     "    keys, inv = np.arange(len(cl)), np.arange(len(cl))",
     T_BB, "test_policy_bootstrap_resamples_whole_clusters"),
    ("BBNV policy breaks not weighted double", BBPO,
     "    return float(np.mean([(1.0 if h > b else -2.0 if h < b else 0.0) for h, b in zip(hits, base)]))",
     "    return float(np.mean([(1.0 if h > b else -1.0 if h < b else 0.0) for h, b in zip(hits, base)]))",
     T_BB, "test_policy_utility_and_outcome"),
    # frozen-gate transfer (22 Oct 2026)
    ("BBNV transfer ceiling against base, not the comparator", BBF,
     '    out["oracle_ceiling_vs_control"] = out["u_oracle"] - out["gates"][CONTROL]["mean_utility_vs_base"]',
     '    out["oracle_ceiling_vs_control"] = out["u_oracle"]',
     T_BB, "test_transfer_summary_reports_the_ceiling_against_the_comparator"),
    ("BBNV transfer stored means ignored", BBF,
     "        z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd]) @ beta",
     "        z = np.hstack([np.ones((len(X), 1)), X / sd]) @ beta",
     T_BB, "test_transfer_frozen_gate_reproduces_the_fitted_gate_without_refitting"),
    ("BBNV transfer threshold 2/3", BBF,
     "        return [bool(p > 0.5) for p in 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))]",
     "        return [bool(p > 2 / 3) for p in 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))]",
     T_BB, "test_transfer_frozen_gate_reproduces_the_fitted_gate_without_refitting"),
    # Phase D analysis and gates (21 Oct 2026)
    ("BBNV analysis F+F' not cyclic", BBA, "\"F+F'\": [\"F%d\" % pair, \"F%d\" % nxt]",
     "\"F+F'\": [\"F%d\" % pair, \"F%d\" % pair]", T_BB, "test_analysis_arms_follow_the_declared_components"),
    ("BBNV analysis breaks not weighted double in utility", BBA,
     "    return 1.0 if d > 0 else -BREAK_WEIGHT if d < 0 else 0.0",
     "    return 1.0 if d > 0 else -1.0 if d < 0 else 0.0", T_BB, "test_analysis_utility_and_folds"),
    ("BBNV analysis gate fitted unweighted", BBA, "    w = np.where(y == 1, 1.0, BREAK_WEIGHT)",
     "    w = np.ones_like(y)", T_BB, "test_analysis_gates_select_C_and_the_strongest_control_on_tune_only"),
    ("BBNV analysis gate threshold 2/3", BBA, "    return (lambda vs: [bool(p > 0.5) for p in predict(_matrix(vs, names))]), params",
     "    return (lambda vs: [bool(p > 2 / 3) for p in predict(_matrix(vs, names))]), params",
     T_BB, "test_analysis_gates_select_C_and_the_strongest_control_on_tune_only"),
    ("BBNV analysis C ties go to the larger C", BBA,
     "    return min(C for C, s in scores.items() if s == best), {str(C): s for C, s in scores.items()}",
     "    return max(C for C, s in scores.items() if s == best), {str(C): s for C, s in scores.items()}",
     T_BB, "test_analysis_gates_select_C_and_the_strongest_control_on_tune_only"),
    ("BBNV analysis same-direction control on one head", BBA, '"Same": ie - sig["I%d" % nxt]',
     '"Same": ie - sig["I%d" % seed]', T_BB, "test_analysis_gate_views_features_and_labels"),
    ("BBNV analysis folds restart per class", BBA, "            folds[j % k].append(i)\n            j += 1\n",
     "            folds[j % k].append(i)\n            j += 1\n        j = 0\n", T_BB, "test_analysis_utility_and_folds"),
    # Phase D scoring and lambda selection (21 Oct 2026)
    ("BBNV scoring banks overlap", BBS,
     '    return {"B_R": pool[perm[:size]], "B_C": pool[perm[size:2 * size]]}',
     '    return {"B_R": pool[perm[:size]], "B_C": pool[perm[:size]]}',
     T_BB, "test_score_banks_are_disjoint_seeded_and_from_negbank"),
    ("BBNV scoring window keeps positives", BBS,
     "    picked = [int(cand_rows[i]) for i in order if int(cand_rows[i]) not in excluded][:k - 1]",
     "    picked = [int(cand_rows[i]) for i in order][:k - 1]",
     T_BB, "test_score_hard_negative_window_excludes_reference_and_positives"),
    ("BBNV scoring reference added twice", BBS,
     "    r_ids = [int(i) for i in bank_r] + ([ref_row] if ref_row not in set(bank_r.tolist()) else [])",
     "    r_ids = [int(i) for i in bank_r] + [ref_row]",
     T_BB, "test_score_forward_window_matches_direct_computation"),
    ("BBNV scoring lambda ties go to the larger value", BBS,
     "    return min(lam for lam, m in means.items() if m == best)",
     "    return max(lam for lam, m in means.items() if m == best)",
     T_BB, "test_score_lambda_selection_takes_the_best_mean_and_the_smaller_tie"),
    ("BBNV scoring smoke heads accepted", BBS,
     '    if res.get("schema") != trn.SCHEMA or res.get("smoke") is not False:',
     '    if res.get("schema") != trn.SCHEMA:',
     T_BB, "test_score_heads_load_only_from_a_full_run_on_these_features"),
    ("BBNV scoring inverse on another universe", BBS,
     "    r_ids = reference_universe(bank_r, ref_row)\n    hyp = hypotheses(head, image_emb[window_rows]",
     "    r_ids = [int(i) for i in bank_r] + [ref_row]\n    hyp = hypotheses(head, image_emb[window_rows]",
     T_BB, "test_score_inverse_window_uses_the_forward_reference_universe"),
    ("BBNV scoring cache not bound to its dump", BBS,
     '    if vman.get("cache_sha256") != prov.sha256_file(cache):', "    if False:",
     T_BB, "test_score_validation_inputs_are_bound_together"),
    ("BBNV scoring window target unchecked", BBS,
     ' or rec["target_gallery_index"] != int(feat["tgt_row"][q]):', ":",
     T_BB, "test_score_validation_inputs_are_bound_together"),
    ("BBNV scoring edit_cos size unchecked", BBS,
     "        if not isinstance(ec, list) or len(ec) != WINDOW or not all(",
     "        if not isinstance(ec, list) or False or not all(",
     T_BB, "test_score_validation_inputs_are_bound_together"),
    # Phase C head training (21 Oct 2026)
    ("BBNV training directions swapped", BBT,
     '    cond, pos = ((feat["ref_row"], feat["tgt_row"]) if direction == "forward"',
     '    cond, pos = ((feat["tgt_row"], feat["ref_row"]) if direction == "forward"',
     T_BB, "test_train_split_directions_and_positive_pool_guard"),
    ("BBNV training positive in pool accepted", BBT,
     "            if ({a, b} | positives.get(a, set()) | positives.get(b, set())) & pool_ids:",
     "            if False:", T_BB, "test_train_split_directions_and_positive_pool_guard"),
    ("BBNV training positive logit unscaled", BBT, "    pos_logit = kappa * (hyp * pos).sum(-1)",
     "    pos_logit = (hyp * pos).sum(-1)", T_BB, "test_train_loss_is_the_m1_mixture_over_the_candidate_set"),
    ("BBNV training negatives with replacement", BBT,
     "    return np.argsort(rng.random((n_rows, pool_size)), axis=1)[:, :n_neg]",
     "    return rng.integers(0, pool_size, size=(n_rows, n_neg))",
     T_BB, "test_train_negative_sampling_is_uniform_without_replacement_and_seeded"),
    ("BBNV training ties go to the larger kappa", BBT,
     '    return min(sorted(runs), key=lambda k: runs[k]["best_dev_nll"])',
     '    return min(sorted(runs, reverse=True), key=lambda k: runs[k]["best_dev_nll"])',
     T_BB, "test_train_selection_and_determinism"),
    # Phase B extraction (20 Oct 2026)
    ("BBNV extraction drift accepted", BBX, "    if bad:\n        raise ExtractError(\"features do not",
     "    if False:\n        raise ExtractError(\"features do not",
     T_BB, "test_extract_reproduction_compares_every_encoded_occurrence"),
    # Phase B extraction audit (20 Oct 2026)
    ("BBNV extraction bare dot product", BBX,
     "    cos = np.einsum(\"nd,nd->n\", a64, b64) / (np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1))",
     "    cos = np.einsum(\"nd,nd->n\", a64, b64)",
     T_BB, "test_extract_reproduction_compares_every_encoded_occurrence"),
    ("BBNV extraction empty comparison accepted", BBX,
     '    if min(errs["n_text_compared"], errs["n_image_occurrences"], errs["n_distinct_images"]) < 1:',
     "    if False:", T_BB, "test_extract_reproduction_compares_every_encoded_occurrence"),
    ("BBNV extraction probe ignores the cached candidates", BBX,
     "    for c in candidate_ids[:n_queries].ravel().tolist():",
     "    for c in range(len(candidate_ids)):",
     T_BB, "test_extract_probe_always_covers_the_probed_queries"),
    ("BBNV extraction candidate range unchecked", BBX,
     "    if cand.min() < 0 or cand.max() >= n_gallery:", "    if False:",
     T_BB, "test_extract_dump_contract_is_validated_before_use"),
    ("BBNV extraction dump norms unchecked", BBX, '    check_unit_rows("c_pool", c_pool, 3)\n', "",
     T_BB, "test_extract_dump_contract_is_validated_before_use"),
    ("BBNV extraction resume reuses another run", BBX,
     "            res.get(\"manifest\", {}).get(\"checkpoint_sha256\")) != (SCHEMA, target, category, checkpoint_sha256):",
     "            res.get(\"manifest\", {}).get(\"checkpoint_sha256\")) == ():",
     T_BB, "test_extract_resume_reuses_only_a_verified_run_of_this_target"),
    ("BBNV extraction caption not aligned", BBX,
     ' or t["caption"] != d["text"]):', "):",
     T_BB, "test_extract_train_alignment_and_strict_inputs"),
    ("BBNV extraction unapproved allocation", BBX, "    if images_sha != APPROVED_IMAGES_SHA256:",
     "    if False:", T_BB, "test_extract_refuses_a_manifest_that_is_not_the_approved_allocation"),
    ("BBNV extraction other inputs accepted", BBX,
     '    if res["manifest"]["inputs_sha256"] != {"captions": prov.sha256_file(cap_path),',
     '    if False and {"captions": prov.sha256_file(cap_path),',
     T_BB, "test_extract_refuses_a_manifest_that_is_not_the_approved_allocation"),
    ("BBNV extraction published without re-load", BBX,
     "    with np.load(path, allow_pickle=False) as z:\n        if sorted(z.files)",
     "    with np.load(path, allow_pickle=True) as z:\n        if sorted(z.files)",
     T_BB, "test_extract_publication_is_reloaded_verified_and_atomic"),
    ("BBNV manifest pool shortfall ignored", BBM,
     "    short = {n: w for n, w in worst.items() if w < N_NEGATIVES}", "    short = {}",
     T_BB, "test_manifest_build_keeps_all_triplets_and_refuses_a_small_pool"),
    ("BBNV negative shortfall shrinks silently", BBP,
     '    if len(eligible) < n:\n        raise PartitionError("only %d eligible negatives after masking, %d requested"\n'
     '                             % (len(eligible), n))\n',
     "    n = min(n, len(eligible))\n",
     T_BB, "test_negative_pool_masks_positives_and_refuses_shortfall"),
    ("BBNV positives not masked from negatives", BBP,
     "    eligible = [i for i in pool if i not in excl]", "    eligible = list(pool)",
     T_BB, "test_negative_pool_masks_positives_and_refuses_shortfall"),
    # BBNV Phase A source audit (19 Oct 2026), one mutant per fixed finding
    ("BBNV prior rank unchecked", BB, "    if a.ndim != 1 or a.shape[0] != n:",
     "    if a.shape[0] != n:", T_BB, "test_audit1_prior_and_table_ranks_are_enforced"),
    ("BBNV window indices coerced", BB, "        if not _is_int(x):", "        if False:",
     T_BB, "test_audit2_window_indices_must_be_genuine_integers"),
    ("BBNV chunk universe order unchecked", BB, "        if tuple(ids) != candidate_ids:",
     "        if False:", T_BB, "test_audit3_chunked_marginal_enforces_the_full_contract"),
    ("BBNV chunk zero marginal returned", BB, "    return _require_finite_marginal(acc)",
     "    return acc", T_BB, "test_audit3_chunked_marginal_enforces_the_full_contract"),
    ("BBNV non-binary labels accepted", BB,
     "    if not np.all(np.isfinite(labf)) or not np.all((labf == 0.0) | (labf == 1.0)):",
     "    if False:", T_BB, "test_audit4_auroc_labels_and_macro_values_are_validated"),
    ("BBNV macro values unvalidated", BB, '_check_auroc_value(x, "macro_auroc")', "x",
     T_BB, "test_audit4_auroc_labels_and_macro_values_are_validated"),
    ("BBNV batched windows accepted", BB, "    if a.ndim != 1 or a.size == 0:",
     "    if a.size == 0:", T_BB, "test_audit5_window_transforms_reject_batches_and_bad_floors"),
    ("BBNV non-positive floor accepted", BB,
     "or not math.isfinite(float(floor)) or float(floor) <= 0.0:",
     "or not math.isfinite(float(floor)):",
     T_BB, "test_audit5_window_transforms_reject_batches_and_bad_floors"),
    ("BBNV mixture report shapes unchecked", BB, "    if comp.shape[0] != log_pi.shape[0]:",
     "    if False:", T_BB, "test_audit6_mixture_report_requires_matching_components"),
    ("BBNV lambda ignored", BB, "    return raw - float(lam) * marg", "    return raw - marg",
     T_BB, "test_lambda_interpolation_spans_raw_and_pmi"),
    ("BBNV padded image ids accepted", BBP,
     "    if not isinstance(image_id, str) or not image_id or image_id != image_id.strip():",
     "    if not isinstance(image_id, str) or not image_id:",
     T_BB, "test_audit10_canonical_image_ids_and_known_positives"),
    ("BBNV known positives not masked", BBP,
     "    mask = {ref, tgt} | set(positives.get(ref, ())) | set(positives.get(tgt, ()))",
     "    mask = {ref, tgt}", T_BB, "test_audit10_canonical_image_ids_and_known_positives"),
    ("BBNV direction typo falls back to inverse", BBO, "    if direction not in DIRECTIONS:",
     "    if False:", T_BB, "test_audit10_oracle_rejects_bad_directions_seeds_and_configs"),
    ("BBNV partial file opened before serialising", BBV,
     "    return _atomic_write_text(path, json.dumps(obj, indent=2, allow_nan=False))",
     '    open(path + ".partial", "w").close()\n'
     "    return _atomic_write_text(path, json.dumps(obj, indent=2, allow_nan=False))",
     T_BB, "test_audit9_atomic_write_leaves_no_partial_and_hashes_match"),
    ("BBNV failed write keeps its partial file", BBV,
     "        if os.path.exists(tmp):\n            os.remove(tmp)\n        raise\n",
     "        raise\n", T_BB, "test_failed_write_after_the_temporary_file_leaves_nothing"),
    ("BBNV run directory may be reused", BBV, "        if os.listdir(path):", "        if False:",
     T_BB, "test_run_directory_is_fresh_complete_and_consistent"),
    ("BBNV verify ignores companion digests", BBV,
     "        if not os.path.exists(p) or sha256_file(p) != sha:",
     "        if not os.path.exists(p):",
     T_BB, "test_run_directory_is_fresh_complete_and_consistent"),
    ("BBNV paired control unvalidated when method undefined", BB,
     '    c_vals = {k: _check_auroc_value(control[k], "control %r" % (k,)) for k in control}',
     "    c_vals = {k: control[k] for k in control}",
     T_BB, "test_paired_auroc_validates_every_value_before_masking"),
    ("control contrast on last permutation", FC,
     "per_query[vname][rname] = median[2]", "per_query[vname][rname] = draws[-1][2]",
     T_FC, "test_control_contrast_uses_the_reported_median_permutation"),
]
# A semantically unchanged edit; its designated test must still pass.
NOOP = ("no-op control (comment only)", FC, "def _is_index(v) -> bool:",
        "def _is_index(v) -> bool:  # no-op control", T_FC,
        "test_validator_rejects_malformed_positions_and_records")

INFRA = ("ImportError", "ModuleNotFoundError", "FileNotFoundError")
RUNNER = r'''
import importlib.util, json, sys, traceback
path, name = sys.argv[1], sys.argv[2]
out = {"status": None, "type": None, "message": None, "traceback": None}
try:
    spec = importlib.util.spec_from_file_location("t_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    getattr(mod, name)()
    out["status"] = "pass"
except BaseException as exc:  # noqa: BLE001 - classified below, never swallowed
    kind = type(exc).__name__
    out.update(type=kind, message=str(exc)[:500], traceback=traceback.format_exc()[-3000:])
    if kind in ("Skipped", "SkipTest"):
        out["status"] = "skip"
    elif kind in %r:
        out["status"] = "infra_error"
    else:
        out["status"] = "fail"
print("@@RESULT@@" + json.dumps(out))
''' % (INFRA,)


def run_test(tree: Path, test_file: str, test_name: str) -> dict:
    """Run one test function in a fresh interpreter against `tree`."""
    proc = subprocess.run([sys.executable, "-c", RUNNER, str(tree / "tests" / test_file),
                           test_name], capture_output=True, text=True, cwd=str(tree))
    for line in proc.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    return {"status": "infra_error", "type": "SubprocessCrash",
            "message": "exit %d" % proc.returncode, "traceback": proc.stderr[-3000:]}


def fresh_tree() -> Path:
    tmp = Path(tempfile.mkdtemp())
    shutil.copytree(ROOT / "scripts", tmp / "scripts")
    shutil.copytree(ROOT / "tests", tmp / "tests",
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    return tmp


def check(label, script, old, new, test_file, test_name) -> dict:
    src = (ROOT / "scripts" / script).read_text(encoding="utf-8")
    if src.count(old) != 1:
        raise SystemExit("mutation anchor for %r not found exactly once in %s" % (label, script))
    tree = fresh_tree()
    try:
        baseline = run_test(tree, test_file, test_name)
        (tree / "scripts" / script).write_text(src.replace(old, new), encoding="utf-8")
        mutant = run_test(tree, test_file, test_name)
    finally:
        shutil.rmtree(tree)
    if baseline["status"] != "pass":
        verdict = "BASELINE-NOT-CLEAN"
    elif mutant["status"] == "fail":
        verdict = "caught"
    elif mutant["status"] == "pass":
        verdict = "SURVIVED"
    else:
        verdict = "UNDETERMINED-" + mutant["status"]
    return {"label": label, "script": script, "test": test_name, "verdict": verdict,
            "baseline": baseline, "mutant": mutant}


def main() -> int:
    results = [check(*m) for m in MUTANTS]
    noop = check(*NOOP)
    for r in results:
        detail = "" if r["verdict"] != "caught" else " (%s)" % r["mutant"]["type"]
        print("%-38s %-20s %s%s" % (r["label"], r["verdict"], r["test"], detail))
    print("%-38s %-20s %s" % (noop["label"],
                              "survived (correct)" if noop["verdict"] == "SURVIVED"
                              else "HARNESS UNRELIABLE: " + noop["verdict"], noop["test"]))
    caught = sum(1 for r in results if r["verdict"] == "caught")
    print("\n%d of %d mutants caught by their designated test; no-op control %s"
          % (caught, len(results), "survived" if noop["verdict"] == "SURVIVED" else "FAILED"))
    report = ROOT / "tests" / "mutation_report.json"
    report.write_text(json.dumps({"mutants": results, "noop_control": noop}, indent=2),
                      encoding="utf-8")
    print("details (types, messages, tracebacks): %s" % report)
    ok = caught == len(results) and noop["verdict"] == "SURVIVED"
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
