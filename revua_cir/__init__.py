"""
ReVUA-CIR: Reverse-Verification, Uncertainty-Aware Composed Image Retrieval.

Reference implementation of the four-module architecture defined in the
paper framework:
    1. Forward Composer         (revua_cir.models.fusion)
    2. Attribute Edit Parser    (revua_cir.models.attribute_parser)
    3. Reverse Verifier         (revua_cir.models.reverse_verifier)
    4. Uncertainty Gate         (revua_cir.models.uncertainty_gate)
tied together by a connective joint loss in revua_cir.models.revua_cir.ReVUACIR.
"""

__version__ = "0.1.0"
