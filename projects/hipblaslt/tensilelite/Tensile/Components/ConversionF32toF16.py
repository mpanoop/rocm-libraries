# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
F32 to F16 Conversion Component

This component handles the conversion of F32 data loaded from global memory
to packed F16 format for computation. It uses v_cvt_pkrtz_f16_f32 instructions
to pack pairs of F32 values into F16 format in VGPRs.
"""

from rocisa.code import Module
from rocisa.container import vgpr
from rocisa.instruction import VCvtPkrtzF16F32, SWaitAlu

from ..Component import Component


class ConvertF32toPackedF16(Component):
    """
    Converts F32 values in VGPRs to packed F16 format.

    Uses v_cvt_pkrtz_f16_f32 instruction which:
    - Takes two F32 inputs (src0, src1)
    - Converts both to F16 with round-to-nearest-zero
    - Packs them into a single 32-bit register (lower/upper 16 bits)
    """

    kernel = {}

    @classmethod
    def matches(cls, writer):
        """Check if this component is needed"""
        kernel = writer.states.kernel
        return (kernel.get("ConvertF32toF16A", False) or
                kernel.get("ConvertF32toF16B", False))

    def __init__(self, writer, tensorChar, numF32Values, f32VgprBase, f16VgprBase):
        """
        Initialize the conversion component.

        Args:
            writer: KernelWriter instance
            tensorChar: 'A' or 'B' to identify which tensor
            numF32Values: Number of F32 values to convert
            f32VgprBase: Base name of F32 VGPR registers (e.g., "G2LAF32")
            f16VgprBase: Base name of destination F16 VGPR registers (e.g., "G2LA")
        """
        self.writer = writer
        self.tensorChar = tensorChar
        self.numF32Values = numF32Values
        self.f32VgprBase = f32VgprBase
        self.f16VgprBase = f16VgprBase

    def __call__(self):
        """
        Generate the conversion instructions.

        Returns:
            Module containing v_cvt_pkrtz_f16_f32 instructions
        """
        module = Module("ConvertF32toF16_%s" % self.tensorChar)
        kernel = self.writer.states.kernel

        # Process F32 values in pairs
        numPairs = self.numF32Values // 2

        for i in range(numPairs):
            f32Idx0 = i * 2
            f32Idx1 = i * 2 + 1
            f16Idx = i

            # v_cvt_pkrtz_f16_f32 dst, src0, src1
            # Converts two F32 to F16 and packs into one VGPR
            module.add(VCvtPkrtzF16F32(
                dst=vgpr("%s+%u" % (self.f16VgprBase, f16Idx)),
                src0=vgpr("%s+%u" % (self.f32VgprBase, f32Idx0)),
                src1=vgpr("%s+%u" % (self.f32VgprBase, f32Idx1)),
                comment="Convert F32 pair [%u:%u] to packed F16" % (f32Idx0, f32Idx1)
            ))

        # Handle odd number of F32 values (convert last one with 0.0)
        if self.numF32Values % 2 == 1:
            lastF32Idx = self.numF32Values - 1
            lastF16Idx = numPairs

            module.add(VCvtPkrtzF16F32(
                dst=vgpr("%s+%u" % (self.f16VgprBase, lastF16Idx)),
                src0=vgpr("%s+%u" % (self.f32VgprBase, lastF32Idx)),
                src1=0.0,  # Pad with zero
                comment="Convert last F32 [%u] to F16 (padded)" % lastF32Idx
            ))

        # Wait for conversion to complete if using expert scheduling
        if kernel.get("ExpertSchedulingMode", 0) > 0:
            module.add(SWaitAlu(va_vdst=0, comment="Wait for F32→F16 conversion"))

        return module

    def getNumF16Vgprs(self):
        """
        Calculate number of F16 VGPRs needed after packing.

        Returns:
            Number of VGPRs needed (half of F32, rounded up)
        """
        return (self.numF32Values + 1) // 2
