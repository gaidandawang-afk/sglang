import unittest

from pydantic import TypeAdapter, ValidationError

from sglang.srt.fault_tolerance.protocol import FaultToleranceApplyRequest


class TestFaultToleranceProtocol(unittest.TestCase):
    def setUp(self):
        self.adapter = TypeAdapter(FaultToleranceApplyRequest)

    def test_valid_requests(self):
        retry = self.adapter.validate_python({"instruction": "retry"})
        scale_down = self.adapter.validate_python(
            {
                "instruction": "scale_down",
                "params": {"removed_dp_ranks": [1]},
                "request_id": "scale-down-1",
            }
        )

        self.assertEqual(retry.request_id, "")
        self.assertEqual(scale_down.params.removed_dp_ranks, [1])

    def test_invalid_requests(self):
        invalid_requests = [
            [],
            {},
            {"instruction": "recover"},
            {"instruction": "retry", "params": []},
            {"instruction": "scale_down", "params": {}},
            {
                "instruction": "scale_down",
                "params": {"removed_dp_ranks": [True]},
            },
            {
                "instruction": "scale_down",
                "params": {"removed_dp_ranks": [-1]},
            },
            {"instruction": "retry", "request_id": 1},
        ]

        for request in invalid_requests:
            with self.subTest(request=request), self.assertRaises(ValidationError):
                self.adapter.validate_python(request)

    def test_invalid_instruction_uses_discriminator_error(self):
        with self.assertRaises(ValidationError) as context:
            self.adapter.validate_python({"instruction": "recover"})

        error = context.exception.errors()[0]
        self.assertEqual(error["type"], "union_tag_invalid")
        self.assertEqual(error["ctx"]["tag"], "recover")
