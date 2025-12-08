/*
754-2008 - IEEE Standard for Floating-Point Arithmetic
(http://www.dsc.ufcg.edu.br/~cnum/modulos/Modulo2/IEEE754_2008.pdf)
*/

#ifndef SPECIAL_CONSTANTS
#define SPECIAL_CONSTANTS

struct SpecialConstants {
    float POS_ZERO;   // +0.0
    float NEG_ZERO;   // -0.0
    float POS_INF;    // +Infinity
    float NEG_INF;    // -Infinity
    float QNAN;       // Canonical quiet NaN
    float NEG_QNAN;   // Canonical quiet NaN (negative sign bit)
    float SNAN;       // Smallest signaling NaN (not guaranteed to trap in GLSL)
    float NEG_SNAN;   // Signaling NaN with negative sign bit
    float TRAP;    // "Trap-like" NaN pattern (all exponent and fraction bits set)
};

const SpecialConstants SPECIAL = SpecialConstants(
    uintBitsToFloat(0x00000000u), // POS_ZERO
    uintBitsToFloat(0x80000000u), // NEG_ZERO
    uintBitsToFloat(0x7f800000u), // POS_INF
    uintBitsToFloat(0xff800000u), // NEG_INF
    uintBitsToFloat(0x7fc00000u), // QNAN
    uintBitsToFloat(0xffc00000u), // NEG_QNAN
    uintBitsToFloat(0x7f800001u), // SNAN
    uintBitsToFloat(0xff800001u), // NEG_SNAN
    uintBitsToFloat(0x7fffffffu)  // TRAP (trap-like: max exponent and all fraction bits)
);
#endif