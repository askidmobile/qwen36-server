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

    def test_cli_rejects_nan_stats(self):
        with tempfile.TemporaryDirectory() as td:
            stats = Path(td) / "stats.txt"
            stats.write_text("0 0 4096 nan 1.0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                alloc.read_stats([stats])


if __name__ == "__main__":
    unittest.main()
