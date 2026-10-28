import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ds41_vram_alloc", ROOT / "scripts" / "qamq" / "ds41_vram_alloc.py"
)
alloc = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = alloc
SPEC.loader.exec_module(alloc)


class Ds41VramAllocTest(unittest.TestCase):
    def setUp(self):
        self.small = alloc.Shape(
            layers=1, experts=2, hidden=16, intermediate=8, vq_dim=8
        )

    def test_matrix_payload_accounting_for_12_and_13_bits(self):
        twelve = alloc.matrix_payload_bytes(8, 16, 12, 8)
        thirteen = alloc.matrix_payload_bytes(8, 16, 13, 8)
        self.assertEqual(twelve % 16, 0)
        self.assertEqual(thirteen % 16, 0)
        self.assertGreater(thirteen, twelve)
        two_bit_delta = alloc.expert_payload_bytes(
            13, self.small
        ) - alloc.expert_payload_bytes(12, self.small)
        self.assertGreater(two_bit_delta, 0)

    def test_mixed_tiers_use_one_nested_codebook_per_layer(self):
        bits = [12, 13]
        self.assertEqual(
            alloc.codebook_bytes(bits, self.small), (1 << 13) * self.small.vq_dim
        )
        self.assertEqual(
            alloc.total_model_bytes(bits, self.small),
            sum(alloc.expert_payload_bytes(b, self.small) for b in bits)
            + (1 << 13) * self.small.vq_dim,
        )

    def test_cache_tiers_fill_vram_then_ram(self):
        shape = alloc.Shape(layers=1, experts=4, hidden=512, intermediate=256, vq_dim=8)
        widths = [12] * 4
        counts = [20, 10, 8, 2]
        ranked = alloc._rank_for_cache(counts, widths, shape, 0)
        one = alloc.expert_payload_bytes(12, shape)
        rows, summary = alloc._select_tiers(ranked, one * 2, one)
        self.assertEqual([row["tier"] for row in rows], ["vram", "vram", "ram", "ssd"])
        self.assertEqual(
            (summary["vram_experts"], summary["ram_experts"], summary["ssd_experts"]),
            (2, 1, 1),
        )
        self.assertAlmostEqual(summary["vram_coverage"], 30 / 40)
        self.assertAlmostEqual(summary["vram_ram_coverage"], 38 / 40)

    def test_runtime_compatibility_rejects_widths_current_v3_cannot_read(self):
        shape = alloc.Shape(layers=2, experts=2, hidden=8, intermediate=8)
        self.assertTrue(alloc.runtime_build_compatibility([12, 12, 13, 13], shape)[0])
        self.assertFalse(alloc.runtime_build_compatibility([12, 13, 12, 12], shape)[0])
        self.assertFalse(alloc.runtime_build_compatibility([11, 11, 12, 12], shape)[0])

    def test_cache_ranking_uses_hits_per_byte_not_raw_hotness(self):
        shape = alloc.Shape(layers=1, experts=2, hidden=512, intermediate=256, vq_dim=8)
        ranked = alloc._rank_for_cache([90, 80], [14, 12], shape, 0)
        self.assertEqual((ranked[0]["expert"], ranked[0]["bits"]), (1, 12))

    def test_profile_round_trip(self):
        ranked = [
            {"layer": 0, "expert": 1},
            {"layer": 0, "expert": 0},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "profile.bin"
            alloc.write_strata_profile(path, ranked, self.small)
            blob = path.read_bytes()
        self.assertEqual(blob[:4], b"STRP")
        ver, layers, experts, slots, total = struct.unpack_from("<5I", blob, 4)
        self.assertEqual((ver, layers, experts, slots, total), (1, 1, 2, 2, 2))
        self.assertEqual(struct.unpack_from("<HH", blob, 24), (0, 1))
        table_off = 24 + 2 * 4
        self.assertEqual(struct.unpack_from("<2i", blob, table_off), (1, 0))

    def test_ds4_profile_is_raw_uint32_counts(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ds4-profile.bin"
            alloc.write_ds4_profile(path, [0, 7], self.small)
            self.assertEqual(struct.unpack("<2I", path.read_bytes()), (0, 7))
            with self.assertRaises(ValueError):
                alloc.write_ds4_profile(path, [-1, 0], self.small)

    def test_uniform_when_errors_are_equal(self):
        curves, _ = alloc.fill_error_curves({}, [10, 11, 12], self.small, 1.0)
        probs = [0.9, 0.1]
        budget = sum(alloc.expert_payload_bytes(12, self.small) for _ in range(2))
        widths = alloc._allocate_nonuniform(
            curves, probs, [10, 11, 12], budget, self.small
        )
        self.assertEqual(widths, [12, 12])

    def test_measured_sensitive_expert_gets_extra_bits(self):
        # Expert 0 is much more sensitive at the reference width.  Its
        # activation-weighted error should win the limited upgrade slots.
        shape = alloc.Shape(layers=1, experts=2, hidden=512, intermediate=256, vq_dim=8)
        measured = {(0, 0): {12: 20.0, 13: 1.0}, (0, 1): {12: 2.0, 13: 1.8}}
        curves, _ = alloc.fill_error_curves(measured, [12, 13], shape, 1.0)
        widths = alloc._allocate_nonuniform(
            curves,
            [0.5, 0.5],
            [12, 13],
            alloc.expert_payload_bytes(12, shape)
            + alloc.expert_payload_bytes(13, shape),
            shape,
        )
        self.assertEqual(widths, [13, 12])

    def test_json_and_strata_trace_parse_to_same_counts(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            json_path = td / "rank.json"
            json_path.write_text(json.dumps([[0, 0, 2], [0, 1, 1]]), encoding="utf-8")
            trace_path = td / "trace.bin"
            record = struct.pack("<ii2i2f", 0, 2, 0, 1, 0.75, 0.25)
            trace_path.write_bytes(record + struct.pack("<ii1i1f", 0, 1, 0, 1.0))
            self.assertEqual(alloc.read_routing(json_path, self.small), [2, 1])
            self.assertEqual(alloc.read_routing(trace_path, self.small), [2, 1])

    def test_ds4_trace_parses_decode_chunk_and_lookahead_records(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ds4-trace.bin"
            decode = bytes((0, 1 | 0x80)) + struct.pack("<2h", 0, 1)
            decode += bytes((1, 0)) + struct.pack("<h", 0)
            chunk = bytes((0x80, 2, 0)) + struct.pack("<4h", 1, 1, 0, 1)
            path.write_bytes(decode + chunk)
            self.assertEqual(
                alloc.read_routing(path, self.small, "ds4-trace", ds4_top_k=2),
                [2, 4],
            )
            self.assertEqual(
                alloc.read_routing(path, self.small, ds4_top_k=2), [2, 4]
            )

    def test_ds4_dump_r_parses_only_the_true_top_k(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ds4-dump-r.bin"
            header = struct.pack("<8q", 0, 0, 1, 2, 2, 1, 1, 3)
            picks = struct.pack("<3i", 0, 1, 0)
            virtual_slots = struct.pack("<2i", 0, 1)
            bounds = struct.pack("<2i", 0, 2)
            row_tokens = struct.pack("<2i", 0, 0)
            logits = struct.pack("<2f", 0.5, 0.5)
            bias = struct.pack("<2f", 0.0, 0.0)
            path.write_bytes(
                header + picks + virtual_slots + bounds + row_tokens + logits + bias
            )
            self.assertEqual(alloc.read_routing(path, self.small, "ds4-dump-r"), [1, 1])
            self.assertEqual(alloc.read_routing(path, self.small), [1, 1])

    def test_cli_writes_ds4_counts_by_default_and_strp_on_request(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            routing = td / "routing.json"
            routing.write_text(json.dumps([[0, 0, 3], [0, 1, 2]]), encoding="utf-8")
            common = [
                "--routing", str(routing), "--layers", "1", "--experts", "2",
                "--hidden", "512", "--intermediate", "256", "--candidates", "12",
                "--out-profile",
            ]
            ds4 = td / "ds4.bin"
            self.assertEqual(alloc.main(common + [str(ds4)]), 0)
            self.assertEqual(struct.unpack("<2I", ds4.read_bytes()), (3, 2))
            stfp = td / "strata.bin"
            self.assertEqual(
                alloc.main(common + [str(stfp), "--profile-format", "strata-ranked"]),
                0,
            )
            self.assertEqual(stfp.read_bytes()[:4], b"STRP")

    def test_routing_only_allocation_is_uniform_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            routing = td / "routing.json"
            routing.write_text(json.dumps([[0, 0, 90], [0, 1, 10]]), encoding="utf-8")
            report = td / "report.json"
            table = td / "nc.txt"
            code = alloc.main(
                [
                    "--routing", str(routing), "--layers", "1", "--experts", "2",
                    "--hidden", "5120", "--intermediate", "2304",
                    "--candidates", "10,11,12,13", "--reference-bits", "12",
                    "--out-table", str(table), "--out-report", str(report),
                ]
            )
            self.assertEqual(code, 0)
            data = json.loads(report.read_text())
            self.assertEqual(data["allocation"]["mode"], "uniform")
            self.assertTrue(data["allocation"]["estimated_error_curve"])
            self.assertFalse(data["allocation"]["evidence"]["ready_for_nonuniform"])
            self.assertTrue(data["allocation"]["safe_for_build"])
            widths = [int(row.split()[2]) for row in table.read_text().splitlines() if not row.startswith("#")]
            self.assertEqual(widths, [4096, 4096])

    def test_routing_only_tiers_require_explicit_experimental_flag(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            routing = td / "routing.json"
            routing.write_text(json.dumps([[0, 0, 90], [0, 1, 10]]), encoding="utf-8")
            report = td / "report.json"
            code = alloc.main(
                [
                    "--routing", str(routing), "--layers", "1", "--experts", "2",
                    "--hidden", "5120", "--intermediate", "2304",
                    "--candidates", "10,11,12,13", "--reference-bits", "12",
                    "--allow-estimated-tiers", "--out-report", str(report),
                ]
            )
            self.assertEqual(code, 0)
            data = json.loads(report.read_text())
            self.assertEqual(data["allocation"]["mode"], "nonuniform")
            self.assertFalse(data["allocation"]["safe_for_build"])

    def test_input_globs_expand_for_windows_cmd(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "stats-1.txt").write_text("0 0 4096 1.0 2.0\n", encoding="utf-8")
            (td / "stats-2.txt").write_text("0 0 8192 0.5 2.0\n", encoding="utf-8")
            paths = alloc.expand_input_paths([str(td / "stats-*.txt")])
            self.assertEqual([p.name for p in paths], ["stats-1.txt", "stats-2.txt"])

    def test_cli_rejects_nan_stats(self):
        with tempfile.TemporaryDirectory() as td:
            stats = Path(td) / "stats.txt"
            stats.write_text("0 0 4096 nan 1.0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                alloc.read_stats([stats])

    def test_placement_detects_uniform_layer_mass(self):
        shape = alloc.Shape(layers=2, experts=2)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            routing = td / "routing.json"
            routing.write_text(json.dumps([[0, 0, 5], [0, 1, 5], [1, 0, 5], [1, 1, 5]]))
            report = td / "r.json"
            self.assertEqual(
                alloc.main([
                    "--routing", str(routing), "--layers", "2", "--experts", "2",
                    "--hidden", "512", "--intermediate", "256", "--candidates", "12",
                    "--mode", "hotness", "--tier-plan", "12,12,12",
                    "--out-report", str(report),
                ]), 0)
            data = json.loads(report.read_text())
            self.assertEqual(data["allocation"]["mode"], "hotness")
            self.assertFalse(data["placement"]["layer_placement_useful"])
            self.assertEqual(data["placement"]["layer_routing_spread"]["std_tokens"], 0.0)
            self.assertGreater(data["placement"]["moe_cache_mib"], 0)
            self.assertIn("--moe-cache-mib", data["placement"]["llama_cpp_command"])

    def test_calibrated_mode_rejects_missing_stats(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            routing = td / "routing.json"
            routing.write_text(json.dumps([[0, 0, 1], [0, 1, 1]]))
            with self.assertRaises(ValueError):
                alloc.main([
                    "--routing", str(routing), "--layers", "1", "--experts", "2",
                    "--hidden", "512", "--intermediate", "256", "--candidates", "12",
                    "--mode", "calibrated", "--out-report", str(td / "r.json"),
                ])

    def test_hotness_tier_plan_puts_hottest_in_vram_and_tail_in_cold(self):
        shape = alloc.Shape(
            layers=1, experts=4, hidden=512, intermediate=256, vq_dim=8
        )
        counts = [90, 8, 1, 1]
        order = sorted(range(4), key=lambda i: (-counts[i] / sum(counts), i))
        self.assertEqual(order[0], 0)
        hot, warm, cold = 12, 10, 9
        self.assertGreater(
            alloc.expert_payload_bytes(hot, shape),
            alloc.expert_payload_bytes(cold, shape),
        )
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            routing = td / "routing.json"
            routing.write_text(
                json.dumps([[0, e, c] for e, c in enumerate(counts)]), encoding="utf-8"
            )
            report = td / "r.json"
            table = td / "nc.txt"
            self.assertEqual(
                alloc.main([
                    "--routing", str(routing), "--layers", "1", "--experts", "4",
                    "--hidden", "512", "--intermediate", "256",
                    "--candidates", "9,10,12", "--min-bits", "9", "--max-bits", "12",
                    "--mode", "hotness", "--tier-plan", "12,10,9",
                    "--vram-mib", "4096", "--dense-mib", "0", "--kv-mib", "0",
                    "--workspace-mib", "0", "--reserve-mib", "0",
                    "--out-table", str(table), "--out-report", str(report),
                ]), 0)
            data = json.loads(report.read_text())
            self.assertEqual(data["allocation"]["mode"], "hotness")
            widths = [int(l.split()[2]) for l in table.read_text().splitlines() if not l.startswith("#")]
            self.assertGreaterEqual(widths[0], widths[-1])
            self.assertIn("placement", data)

    def test_vq_error_shrinks_monotonically_with_width(self):
        import numpy as np

        spec = importlib.util.spec_from_file_location(
            "vq_error", ROOT / "scripts" / "qamq" / "vq_error.py"
        )
        vq = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = vq
        spec.loader.exec_module(vq)

        rng = np.random.default_rng(5)
        matrix = rng.standard_normal((256, 256)).astype(np.float32)
        errors = [
            vq.vq_error(matrix, 8, bits, blocks=4096, iters=4, seed=1)
            for bits in (8, 10, 12)
        ]
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertGreater(errors[0], 0.0)

    def test_dsx_activation_stats_are_rejected_with_a_pointer(self):
        with tempfile.TemporaryDirectory() as td:
            stats = Path(td) / "L00.json"
            stats.write_text('{"experts": {"0": {"n_eval": 8, "e_q2_0": 0.1}}}')
            with self.assertRaises(ValueError) as ctx:
                alloc.read_stats([stats])
            self.assertIn("dsx", str(ctx.exception))


    def test_dqvp_container_round_trips_mixed_widths(self):
        import importlib.util as iu

        spec = iu.spec_from_file_location(
            "dqvp_pack", ROOT / "scripts" / "qamq" / "dqvp_pack.py"
        )
        dq = iu.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = dq
        spec.loader.exec_module(dq)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pack.bin"
            info = dq.self_check(path, blocks=32)
            self.assertEqual(info["verified_slots"], 6)
            got = dq.read_container(path)
        self.assertEqual(got["layers"], 2)
        self.assertEqual(sorted({c.nc for c in got["codecs"]}), [9, 10, 12, 13])

    def test_dqvp_bit_packing_matches_the_byte_count(self):
        import importlib.util as iu

        spec = iu.spec_from_file_location(
            "dqvp_pack2", ROOT / "scripts" / "qamq" / "dqvp_pack.py"
        )
        dq = iu.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = dq
        spec.loader.exec_module(dq)

        for nc in (9, 10, 12, 13):
            values = list(range(1 << nc))
            blob = dq._pack_indices(values, nc, 8)
            self.assertEqual(len(blob), (len(values) * nc + 7) // 8)
            self.assertEqual(dq._unpack_indices(blob, len(values), nc), values)

    def test_dense_budget_reproduces_the_downloaded_shard_size(self):
        import importlib.util as iu

        spec = iu.spec_from_file_location(
            "dense_budget", ROOT / "scripts" / "qamq" / "dense_budget.py"
        )
        db = iu.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = db
        spec.loader.exec_module(db)

        shape = db.shapes()
        self.assertEqual(len(shape), 1042)
        # expert payload: 40 layers x 3 matrices x 384 experts x hidden x 2304
        self.assertEqual(shape["blk.0.ffn_up_exps.weight"], (5120, 384 * 2304))
        # hyper-connection fan-out is stored, not a scalar
        self.assertEqual(shape["blk.0.hc_attn_fn.weight"], (20480, 24))
        # a real f32 tensor is 4 bytes per element
        self.assertEqual(db.nbytes((5120,), "F32"), 5120 * 4)
        self.assertEqual(db.nbytes((5120, 384), "BF16"), 5120 * 384 * 2)


if __name__ == "__main__":
    unittest.main()
