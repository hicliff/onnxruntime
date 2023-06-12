// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Portions of this code are from System.Half struct dotnet runtime.
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    internal class BitOps
    {
        /// <summary>
        /// Required because BitOperations are not available in NETSTANDARD 2.0.
        /// There are more efficient ways with bit twiddling, but this one has clarity.
        /// </summary>
        /// <param name="num">value</param>
        /// <returns>number of leading zeros. Useful to compute log2 as well.</returns>
        internal static int LeadingZeroCount(uint num)
        {
            if (num == 0)
            {
                return 32;
            }

            int count = 0;
            while ((num & 0xF000_0000) == 0)
            {
                count += 4;
                num <<= 4;
            }

            while ((num & 0x8000_0000) == 0)
            {
                count += 1;
                num <<= 1;
            }
            return count;
        }

        /// <summary>
        /// Needed because BitConverter impl is not available until
        /// later versions. Assumes that the bits are constructed in a proper
        /// 
        /// The alternatives as BitConverter.GetBytes(src) and then BitConverter.ToSingle()
        /// produces extra bytes array which may be too much for the amount of elements in a tensor.
        /// layout.
        /// </summary>
        /// <param name="singleBits"></param>
        /// <returns></returns>
        internal static float UInt32BitsToSingle(uint singleBits)
        {
            float result;
            unsafe
            {
                Buffer.MemoryCopy(&singleBits, &result, sizeof(uint), sizeof(uint));
            }
            return result;
        }

        internal static uint SingleToUInt32Bits(float single)
        {
            uint result;
            unsafe
            {
                Buffer.MemoryCopy(&single, &result, sizeof(uint), sizeof(uint));
            }
            return result;  
        }
    }


    /// <summary>
    /// This value type represents A Float16 value
    /// it is blittable as defined in https://docs.microsoft.com/en-us/dotnet/framework/interop/blittable-and-non-blittable-types
    /// and as such, represented the same way in managed and native memories. This means that arrays of this type
    /// do not have to be copied to be passed to native memory but simply pinned and read by native code. Thus,
    /// one can create a Tensor on top of an array of these structures and feed it directly to Onnxruntime library.
    /// Binary wise, it is the same as ushort[] (uint16_t in C++). However, we would like a separate type for type dispatching.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct Float16 : 
        IComparable,
        IComparable<Float16>,
        IEquatable<Float16>
    {
        internal const ushort SignMask = 0x8000;
        internal const int SignShift = 15;
        internal const byte ShiftedSignMask = SignMask >> SignShift;

        internal const ushort BiasedExponentMask = 0x7C00;
        internal const int BiasedExponentShift = 10;
        internal const byte ShiftedBiasedExponentMask = BiasedExponentMask >> BiasedExponentShift;

        internal const ushort TrailingSignificandMask = 0x03FF;

        internal const byte MinSign = 0;
        internal const byte MaxSign = 1;

        internal const byte MinBiasedExponent = 0x00;
        internal const byte MaxBiasedExponent = 0x1F;

        internal const byte ExponentBias = 15;

        internal const sbyte MinExponent = -14;
        internal const sbyte MaxExponent = +15;

        internal const ushort MinTrailingSignificand = 0x0000;
        internal const ushort MaxTrailingSignificand = 0x03FF;

        // Constants representing the private bit-representation for various default values

        private const ushort PositiveZeroBits = 0x0000;
        private const ushort NegativeZeroBits = 0x8000;

        private const ushort EpsilonBits = 0x0001;

        private const ushort PositiveInfinityBits = 0x7C00;
        private const ushort NegativeInfinityBits = 0xFC00;

        private const ushort PositiveQNaNBits = 0x7E00;
        private const ushort NegativeQNaNBits = 0xFE00;

        private const ushort MinValueBits = 0xFBFF;
        private const ushort MaxValueBits = 0x7BFF;

        private const ushort PositiveOneBits = 0x3C00;
        private const ushort NegativeOneBits = 0xBC00;

        private const ushort EBits = 0x4170;
        private const ushort PiBits = 0x4248;
        private const ushort TauBits = 0x4648;

        // Well-defined and commonly used values

        public static Float16 Epsilon => new Float16(EpsilonBits);                        //  5.9604645E-08

        public static Float16 PositiveInfinity => new Float16(PositiveInfinityBits);      //  1.0 / 0.0;

        public static Float16 NegativeInfinity => new Float16(NegativeInfinityBits);      // -1.0 / 0.0

        public static Float16 NaN => new Float16(NegativeQNaNBits);                       //  0.0 / 0.0

        /// <inheritdoc cref="IMinMaxValue{TSelf}.MinValue" />
        public static Float16 MinValue => new Float16(MinValueBits);                      // -65504

        /// <inheritdoc cref="IMinMaxValue{TSelf}.MaxValue" />
        public static Float16 MaxValue => new Float16(MaxValueBits);                      //  65504

        /// <summary>
        /// float16 representation bits
        /// </summary>
        public readonly ushort value;

        /// <summary>
        /// Ctor from ushort bits, no conversion is done
        /// </summary>
        /// <param name="v"></param>
        public Float16(ushort v)
        {
            value = v;
        }

        private Float16(bool sign, ushort exp, ushort sig) => 
            value = (ushort)(((sign ? 1 : 0) << SignShift) + (exp << BiasedExponentShift) + sig);

        internal byte BiasedExponent
        {
            get
            {
                ushort bits = value;
                return ExtractBiasedExponentFromBits(bits);
            }
        }

        internal sbyte Exponent
        {
            get
            {
                return (sbyte)(BiasedExponent - ExponentBias);
            }
        }

        internal ushort Significand
        {
            get
            {
                return (ushort)(TrailingSignificand | ((BiasedExponent != 0) ? (1U << BiasedExponentShift) : 0U));
            }
        }

        internal ushort TrailingSignificand
        {
            get
            {
                ushort bits = value;
                return ExtractTrailingSignificandFromBits(bits);
            }
        }

        internal static byte ExtractBiasedExponentFromBits(ushort bits)
        {
            return (byte)((bits >> BiasedExponentShift) & ShiftedBiasedExponentMask);
        }

        internal static ushort ExtractTrailingSignificandFromBits(ushort bits)
        {
            return (ushort)(bits & TrailingSignificandMask);
        }

        /// <inheritdoc cref="IComparisonOperators{TSelf, TOther, TResult}.op_LessThan(TSelf, TOther)" />
        public static bool operator <(Float16 left, Float16 right)
        {
            if (IsNaN(left) || IsNaN(right))
            {
                // IEEE defines that NaN is unordered with respect to everything, including itself.
                return false;
            }

            bool leftIsNegative = IsNegative(left);

            if (leftIsNegative != IsNegative(right))
            {
                // When the signs of left and right differ, we know that left is less than right if it is
                // the negative value. The exception to this is if both values are zero, in which case IEEE
                // says they should be equal, even if the signs differ.
                return leftIsNegative && !AreZero(left, right);
            }

            return (left.value != right.value) && ((left.value < right.value) ^ leftIsNegative);
        }

        /// <inheritdoc cref="IComparisonOperators{TSelf, TOther, TResult}.op_GreaterThan(TSelf, TOther)" />
        public static bool operator >(Float16 left, Float16 right)
        {
            return right < left;
        }

        /// <inheritdoc cref="IComparisonOperators{TSelf, TOther, TResult}.op_LessThanOrEqual(TSelf, TOther)" />
        public static bool operator <=(Float16 left, Float16 right)
        {
            if (IsNaN(left) || IsNaN(right))
            {
                // IEEE defines that NaN is unordered with respect to everything, including itself.
                return false;
            }

            bool leftIsNegative = IsNegative(left);

            if (leftIsNegative != IsNegative(right))
            {
                // When the signs of left and right differ, we know that left is less than right if it is
                // the negative value. The exception to this is if both values are zero, in which case IEEE
                // says they should be equal, even if the signs differ.
                return leftIsNegative || AreZero(left, right);
            }

            return (left.value == right.value) || ((left.value < right.value) ^ leftIsNegative);
        }

        /// <inheritdoc cref="IComparisonOperators{TSelf, TOther, TResult}.op_GreaterThanOrEqual(TSelf, TOther)" />
        public static bool operator >=(Float16 left, Float16 right)
        {
            return right <= left;
        }

        /// <summary>
        /// Compares values of two Float16 for binary equality
        /// </summary>
        /// <param name="left">left hand side</param>
        /// <param name="right">right hand side</param>
        /// <returns>result of value comparisons</returns>
        public static bool operator ==(Float16 left, Float16 right) 
        {
            if (IsNaN(left) || IsNaN(right))
            {
                // IEEE defines that NaN is not equal to anything, including itself.
                return false;
            }

            return left.value == right.value;
        }

        /// <summary>
        /// Compares values of two Float16 for binary inequality
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <returns>result of value comparisons</returns>
        public static bool operator !=(Float16 lhs, Float16 rhs) 
        {
            return !(lhs == rhs);
        }


        /// <summary>
        /// Determines whether the specified value is finite (zero, subnormal, or normal).
        /// </summary>
        /// <param name="value">Float16 instance.</param>
        /// <returns>true if the value is finite</returns>
        public static bool IsFinite(Float16 value)
        {
            return StripSign(value) < PositiveInfinityBits;
        }

        /// <summary>
        /// Determines whether the specified value is infinite.
        /// </summary>
        /// <param name="value">Float16 instance.</param>
        /// <returns>true if the value is infinite</returns>
        public static bool IsInfinity(Float16 value)
        {
            return StripSign(value) == PositiveInfinityBits;
        }

        /// <summary>
        /// Determines whether the specified value is NaN.
        /// </summary>
        /// 
        /// <param name="value">Float16 instance</param>
        /// <returns>true if the value is not a number</returns>
        public static bool IsNaN(Float16 value)
        {
            return StripSign(value) > PositiveInfinityBits;
        }

        /// <summary>
        /// Determines whether the specified value is negative.
        /// </summary>
        /// <param name="value">Float16 instance</param>
        /// <returns>true if the value is negative</returns></returns>
        public static bool IsNegative(Float16 value)
        {
            return (short)(value.value) < 0;
        }

        /// <summary>
        /// Determines whether the specified value is negative infinity.
        /// </summary>
        /// 
        /// <param name="value">Float16 instance</param>
        /// <returns>true if the value is negative infinity</returns>
        public static bool IsNegativeInfinity(Float16 value)
        {
            return value.value == NegativeInfinityBits;
        }

        /// <summary>Determines whether the specified value is normal.</summary>
        public static bool IsNormal(Float16 value)
        {
            uint absValue = StripSign(value);
            return (absValue < PositiveInfinityBits)    // is finite
                && (absValue != 0)                      // is not zero
                && ((absValue & BiasedExponentMask) != 0);    // is not subnormal (has a non-zero exponent)
        }

        /// <summary>Determines whether the specified value is positive infinity.</summary>
        public static bool IsPositiveInfinity(Float16 value)
        {
            return value.value == PositiveInfinityBits;
        }

        /// <summary>Determines whether the specified value is subnormal.</summary>
        // This is probably not worth inlining, it has branches and should be rarely called
        public static bool IsSubnormal(Float16 value)
        {
            uint absValue = StripSign(value);
            return (absValue < PositiveInfinityBits)    // is finite
                && (absValue != 0)                      // is not zero
                && ((absValue & BiasedExponentMask) == 0);    // is subnormal (has a zero exponent)
        }

        private static bool AreZero(Float16 left, Float16 right)
        {
            // IEEE defines that positive and negative zero are equal, this gives us a quick equality check
            // for two values by or'ing the private bits together and stripping the sign. They are both zero,
            // and therefore equivalent, if the resulting value is still zero.
            return (ushort)((left.value | right.value) & ~SignMask) == 0;
        }

        private static bool IsNaNOrZero(Float16 value)
        {
            return ((value.value - 1) & ~SignMask) >= PositiveInfinityBits;
        }

        private static uint StripSign(Float16 value)
        {
            return (ushort)(value.value & ~SignMask);
        }

 
        /// <summary>
        /// Compares this object to another object, returning an integer that indicates the relationship.
        /// </summary>
        /// 
        /// <param name="obj">Object to compare to</param>
        /// <returns>A value less than zero if this is less than <paramref name="obj"/>,
        /// zero if this is equal to <paramref name="obj"/>, or a value greater than zero
        /// if this is greater than <paramref name="obj"/>.
        /// </returns>
        /// <exception cref="ArgumentException">Thrown when <paramref name="obj"/> is not of type <see cref="Float16"/>.</exception>
        public int CompareTo(object obj)
        {
            if (!(obj is Float16))
            {
                return (obj is null) ? 1 : throw new ArgumentException("Object must be of type Float16");
            }
            return CompareTo((Float16)(obj));
        }

        /// <summary>
        /// Compares this object to another object, returning an integer that indicates the relationship.
        /// </summary>
        /// <param name="other">Object to compare to</param>
        /// <returns>A value less than zero if this is less than <paramref name="other"/>,
        /// zero if this is equal to <paramref name="other"/>, 
        /// or a value greater than zero if this is greater than <paramref name="other"/>.</returns>
        public int CompareTo(Float16 other)
        {
            if (this < other)
            {
                return -1;
            }

            if (this > other)
            {
                return 1;
            }

            if (this == other)
            {
                return 0;
            }

            if (IsNaN(this))
            {
                return IsNaN(other) ? 0 : -1;
            }

            Debug.Assert(IsNaN(other));
            return 1;
        }

        /// <summary>
        /// Returns a value indicating whether this instance and other Float16 represent the same value.
        /// </summary>
        /// <param name="other">A Float16 object to compare to this instance.</param>
        /// <returns>true if other.value is equal to this instance; otherwise, false.</returns>
        public bool Equals(Float16 other)
        {
            return value == other.value
                || AreZero(this, other)
                || (IsNaN(this) && IsNaN(other));
        }

        /// <summary>
        /// Returns a value indicating whether this instance and a specified System.Object
        /// represent the same type and value.
        /// </summary>
        /// <param name="obj">An System.Object.</param>
        /// <returns>true if obj is Float16 and its value is equal to this instance; otherwise, false.</returns>
        public override bool Equals(object obj)
        {
            return (obj is Float16 other) && Equals(other);
        }

        /// <summary>
        /// Returns the hash code for this instance.
        /// </summary>
        /// <returns>A 32-bit signed integer hash code.</returns>
        public override int GetHashCode()
        {
            if (IsNaNOrZero(this))
            {
                // All NaNs should have the same hash code, as should both Zeros.
                return value & PositiveInfinityBits;
            }
            return value;
        }

        /// <summary>
        /// Returns a string representation of the current value.
        /// </summary>
        public override string ToString()
        {
            return $"{value} : {(float)this}";
        }

        /// <summary>
        /// Explicit conversion
        /// </summary>
        /// <returns></returns>
        public float ToFloat()
        {
            return (float)this;
        }

        /// <summary>
        /// Converts a 16-bit unsigned integer to a float and then to Float16
        /// </summary>
        /// <param name="value">A 16-bit unsigned integer.</param>
        /// <returns>A Float16 that represents the converted 16-bit unsigned integer.</returns>
        public static explicit operator Float16(ushort value) => (Float16)(float)value;

        /// <summary>Explicitly converts a half-precision floating-point value to its nearest representable <see cref="ushort" /> value.</summary>
        /// <param name="value">The value to convert.</param>
        /// <returns><paramref name="value" /> converted to its nearest representable <see cref="ushort" /> value.</returns>
        public static explicit operator ushort(Float16 value) => (ushort)(float)value;

        // Lifted from .NET source code internal code
        // Constants for Single precision format
        internal const uint SingleBiasedExponentMask = 0x7F80_0000;
        internal const int SingleBiasedExponentShift = 23;

        internal const uint SingleSignMask = 0x8000_0000;
        internal const int SingleSignShift = 31;

        internal const uint SingleTrailingSignificandMask = 0x007F_FFFF;

        /// <summary>Explicitly converts a <see cref="float" /> value to its nearest representable half-precision floating-point value.</summary>
        /// <param name="value">The value to convert.</param>
        /// <returns><paramref name="value" /> converted to its nearest representable half-precision floating-point value.</returns>
        public static explicit operator Float16(float value)
        {
            const int SingleMaxExponent = 0xFF;

            uint floatInt = BitOps.SingleToUInt32Bits(value);
            bool sign = (floatInt & SingleSignMask) >> SingleSignShift != 0;
            int exp = (int)(floatInt & SingleBiasedExponentMask) >> SingleBiasedExponentShift;
            uint sig = floatInt & SingleTrailingSignificandMask;

            if (exp == SingleMaxExponent)
            {
                if (sig != 0) // NaN
                {
                    return CreateFloat16NaN(sign, (ulong)sig << 41); // Shift the significand bits to the left end
                }
                return sign ? NegativeInfinity : PositiveInfinity;
            }

            uint sigHalf = sig >> 9 | ((sig & 0x1FFU) != 0 ? 1U : 0U); // RightShiftJam

            if ((exp | (int)sigHalf) == 0)
            {
                return new Float16(sign, 0, 0);
            }

            return new Float16(RoundPackToFloat16(sign, (short)(exp - 0x71), (ushort)(sigHalf | 0x4000)));
        }

        /// <summary>Explicitly converts a half-precision floating-point value to its nearest representable <see cref="float" /> value.</summary>
        /// <param name="value">The value to convert.</param>
        /// <returns><paramref name="value" /> converted to its nearest representable <see cref="float" /> value.</returns>
        public static explicit operator float(Float16 value)
        {
            bool sign = IsNegative(value);
            int exp = value.BiasedExponent;
            uint sig = value.TrailingSignificand;

            if (exp == MaxBiasedExponent)
            {
                if (sig != 0)
                {
                    return CreateSingleNaN(sign, (ulong)sig << 54);
                }
                return sign ? float.NegativeInfinity : float.PositiveInfinity;
            }

            if (exp == 0)
            {
                if (sig == 0)
                {
                    return BitOps.UInt32BitsToSingle(sign ? SingleSignMask : 0); // Positive / Negative zero
                }
                (exp, sig) = NormSubnormalF16Sig(sig);
                exp -= 1;
            }

            return CreateSingle(sign, (byte)(exp + 0x70), sig << 13);
        }

        // IEEE 754 specifies NaNs to be propagated
        public static Float16 Negate(Float16 value)
        {
            return IsNaN(value) ? value : new Float16((ushort)(value.value ^ SignMask));
        }

        private static (int Exp, uint Sig) NormSubnormalF16Sig(uint sig)
        {
            int shiftDist = BitOps.LeadingZeroCount(sig) - 16 - 5;
            return (1 - shiftDist, sig << shiftDist);
        }

        #region Utilities

        // Significand bits should be shifted towards to the left end before calling these methods
        // Creates Quiet NaN if significand == 0
        private static Float16 CreateFloat16NaN(bool sign, ulong significand)
        {
            const ushort NaNBits = BiasedExponentMask | 0x200; // Most significant significand bit

            uint signInt = (sign ? 1U : 0U) << SignShift;
            ushort sigInt = (ushort)(significand >> 54);

            ushort ushortBits = (ushort)(signInt | NaNBits | sigInt);
            return new Float16(ushortBits);
        }

        private static ushort RoundPackToFloat16(bool sign, short exp, ushort sig)
        {
            const int RoundIncrement = 0x8; // Depends on rounding mode but it's always towards closest / ties to even
            int roundBits = sig & 0xF;

            if ((uint)exp >= 0x1D)
            {
                if (exp < 0)
                {
                    sig = (ushort)ShiftRightJam(sig, -exp);
                    exp = 0;
                    roundBits = sig & 0xF;
                }
                else if (exp > 0x1D || sig + RoundIncrement >= 0x8000) // Overflow
                {
                    return sign ? NegativeInfinityBits : PositiveInfinityBits;
                }
            }

            sig = (ushort)((sig + RoundIncrement) >> 4);
            sig &= (ushort)~(((roundBits ^ 8) != 0 ? 0 : 1) & 1);

            if (sig == 0)
            {
                exp = 0;
            }

            return new Float16(sign, (ushort)exp, sig).value;
        }

        // If any bits are lost by shifting, "jam" them into the LSB.
        // if dist > bit count, Will be 1 or 0 depending on i
        // (unlike bitwise operators that masks the lower 5 bits)
        private static uint ShiftRightJam(uint i, int dist) => dist < 31 ? (i >> dist) | (i << (-dist & 31) != 0 ? 1U : 0U) : (i != 0 ? 1U : 0U);

        private static ulong ShiftRightJam(ulong l, int dist) => dist < 63 ? (l >> dist) | (l << (-dist & 63) != 0 ? 1UL : 0UL) : (l != 0 ? 1UL : 0UL);

        private static float CreateSingleNaN(bool sign, ulong significand)
        {
            const uint NaNBits = SingleBiasedExponentMask | 0x400000; // Most significant significand bit

            uint signInt = (sign ? 1U : 0U) << SingleSignShift;
            uint sigInt = (uint)(significand >> 41);
            uint singleBits = signInt | NaNBits | sigInt;

            return BitOps.UInt32BitsToSingle(singleBits);
        }

        private static float CreateSingle(bool sign, byte exp, uint sig)
        {
            uint signInt = (sign ? 1U : 0U) << SingleSignShift;
            uint expInt = ((uint)exp << SingleBiasedExponentShift) + sig;
            uint singleBits = signInt + expInt;

            return BitOps.UInt32BitsToSingle(singleBits);
        }

        #endregion
    }

    /// <summary>
    /// This value type represents A BFloat16 value
    /// it is blittable as defined in https://docs.microsoft.com/en-us/dotnet/framework/interop/blittable-and-non-blittable-types
    /// and as such, represented the same way in managed and native memories. This means that arrays of this type
    /// do not have to be copied to be passed to native memory but simply pinnned and read by native code. Thus,
    /// one can create a Tensor on top of an array of these structures and feed it directly to Onnxruntime library.
    /// Binary wise, it is the same as ushort[] (uint16_t in C++). However, we would like a separate type for type dispatching.
    /// </summary>
    public struct BFloat16
    {
        /// <summary>
        /// bfloat16 representation bits
        /// </summary>
        public ushort value;
        /// <summary>
        /// Ctor
        /// </summary>
        /// <param name="v"></param>
        public BFloat16(ushort v)
        {
            value = v;
        }
        /// <summary>
        /// Converts to ushort
        /// </summary>
        /// <param name="bf">instance of BFloat16</param>
        /// <returns>value member</returns>
        public static implicit operator ushort(BFloat16 bf) { return bf.value; }
        /// <summary>
        /// Converts a 16-bit unsigned integer to a BFloat16.
        /// </summary>
        /// <param name="value">A 16-bit unsigned integer.</param>
        /// <returns>A BFloat16 that represents the converted 16-bit unsigned integer.</returns>
        public static implicit operator BFloat16(ushort value) { return new BFloat16(value); }
        /// <summary>
        /// Compares values of two BFloat16 for binary equality
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <returns>result of value comparisons</returns>
        public static bool operator ==(BFloat16 lhs, BFloat16 rhs) { return lhs.value == rhs.value; }
        /// <summary>
        /// Compares values of two BFloat16 for binary inequality
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <returns>result of value comparisons</returns>
        public static bool operator !=(BFloat16 lhs, BFloat16 rhs) { return lhs.value != rhs.value; }

        /// <summary>
        /// Returns a value indicating whether this instance and other BFloat16 represent the same value.
        /// </summary>
        /// <param name="other">A BFloat16 object to compare to this instance.</param>
        /// <returns>true if other.value is equal to this instance; otherwise, false.</returns>
        public bool Equals(BFloat16 other)
        {
            return (other == this);
        }

        /// <summary>
        /// Returns a value indicating whether this instance and a specified System.Object
        /// represent the same type and value.
        /// </summary>
        /// <param name="obj">An System.Object.</param>
        /// <returns>true if obj is BFloat16 its value is equal to this instance; otherwise, false.</returns>
        public override bool Equals(object obj)
        {
            bool result = false;
            if (obj is BFloat16)
            {
                BFloat16 bfl16 = (BFloat16)obj;
                result = (bfl16 == this);
            }
            return result;
        }
        /// <summary>
        /// Returns the hash code for this instance.
        /// </summary>
        /// <returns>A 32-bit signed integer hash code.</returns>
        public override int GetHashCode()
        {
            return value.GetHashCode();
        }
    }
}