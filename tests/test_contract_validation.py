"""Boundary tests for financial-contract and numerical reference validation.

These checks complement the pricing tests: malformed inputs should fail at
construction rather than propagating NaNs or singular matrices into results.
"""
import unittest
from dataclasses import replace

from stopping import Option, validation_put, crr_bermudan_put, convolution_bermudan_put


class ContractValidationTests(unittest.TestCase):
    def test_invalid_option_parameters(self):
        for changes in (
            {"d": 0}, {"d": 1.5}, {"d": True}, {"dates": 0},
            {"dates": 2.5}, {"s0": 0}, {"strike": -1},
            {"maturity": 0}, {"sigma": -0.2},
            {"rate": float("nan")}, {"dividend": float("inf")},
            {"rho": float("nan")},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(Option(), **changes)

    def test_equicorrelation_domain(self):
        for rho in (-0.5, 1.0):
            with self.subTest(rho=rho), self.assertRaises(ValueError):
                Option(kind="max_call", d=3, rho=rho)
        Option(kind="max_call", d=3, rho=0.3)

    def test_reference_method_input_validation(self):
        put = validation_put()
        for steps in (0, -3, 1.5):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                crr_bermudan_put(put, steps)
        for spacing in (0, -0.1, float("nan"), float("inf")):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                convolution_bermudan_put(put, spacing)
        with self.assertRaises(ValueError):
            crr_bermudan_put(replace(put, sigma=0))
        with self.assertRaises(ValueError):
            convolution_bermudan_put(replace(put, sigma=0))


if __name__ == "__main__":
    unittest.main()
