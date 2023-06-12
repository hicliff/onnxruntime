// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tests
{
    [Collection("Ort Floating Point 16 tests")]
    public class OrtFloat16Tests
    {
        [Fact(DisplayName = "ConvertShortToFloat16")]
        public void ConvertShortToFloat16()
        {
            ushort[] shortValues = { 0, 1, 2, 3, 4, 5, 6, 7 };
            float[] floatValues = { 0, 1, 2, 3, 4, 5, 6, 7 };
            for (int i = 0; i < shortValues.Length; i++)
            {
                Float16 v = (Float16)shortValues[i];
                Assert.Equal(shortValues[i], (ushort)v);
                Assert.Equal(floatValues[i], (float)v);
                Assert.Equal(floatValues[i], v.ToFloat());
            }
        }

        [Fact(DisplayName = "ConvertFloatToFloat16")]
        public void ConvertFloatToFloat16()
        {
            float[] floatValues = { 0, 1, 2, 3, 4, 5, 6, 7 };
            for (int i = 0; i < floatValues.Length; i++)
            {
                Float16 v = (Float16)floatValues[i];
                Assert.Equal(floatValues[i], (float)v);
                Assert.Equal(floatValues[i], v.ToFloat());
            }
        }

        [Fact(DisplayName = "TestComparisionOperators")]
        public void TestComparisionOperators()
        {
            Float16 left = (Float16)(float)-33.33f;
            Float16 leftSame = (Float16)(float)-33.33f;
            Float16 right = (Float16)(float)66.66f;
            Float16 rightSame = (Float16)(float)66.66f;

            Assert.True(right > Float16.Epsilon);

            Assert.True(left == leftSame);
            Assert.False(left == Float16.Negate(leftSame));

            Assert.True(right == rightSame);
            Assert.False(right == Float16.Negate(rightSame));

            Assert.True(left < right);
            Assert.True(left > Float16.Negate(right));
            Assert.True(Float16.Negate(left) < right);

            Assert.True(left <= right);
            Assert.True(left >= Float16.Negate(right));
            Assert.False(left > right);
            Assert.False(left >= right);
            Assert.True(Float16.Negate(left) <= right);
            Assert.False(left == right);
            Assert.False(right == left);
            Assert.True(left != right);
            Assert.True(right != left);
        }

        [Fact(DisplayName = "TestNAN")]
        public void TestNAN()
        {
            Float16 fp16NANFromSingle = (Float16)float.NaN;
            Assert.True(Float16.IsNaN(fp16NANFromSingle));
            Assert.Equal(Float16.NaN, fp16NANFromSingle);

            float NanFromFloat16 = fp16NANFromSingle.ToFloat();
            Assert.True(float.IsNaN(NanFromFloat16));

            // IEqualityComparable returns true, because it tests
            // objects, not numbers.
            Assert.Equal(fp16NANFromSingle, Float16.NaN);

            Assert.Equal(Float16.NaN, Float16.Negate(Float16.NaN));
        }

        [Fact(DisplayName = "TestNANComparision")]
        public void TestNANComparisionOperators()
        {
            // NaN is not ordered with respect to anything
            // including itself

            // IEqualityComparable returns true, because it tests
            // objects, not numbers.
            Assert.Equal(Float16.NaN, Float16.NaN);
            Assert.False(Float16.NaN < Float16.NaN);
            Assert.False(Float16.NaN > Float16.NaN);
            Assert.False(Float16.NaN <= Float16.NaN);
            Assert.False(Float16.NaN >= Float16.NaN);
            Assert.False(Float16.NaN == Float16.NaN);

            // IEqualityComparable returns false, because it tests
            // objects, not numbers.
            Assert.NotEqual(Float16.NaN, Float16.MaxValue);

            Assert.False(Float16.NaN < Float16.MaxValue);
            Assert.False(Float16.MaxValue < Float16.NaN);
            Assert.False(Float16.NaN == Float16.MaxValue);
            Assert.False(Float16.MaxValue == Float16.NaN);
            Assert.False(Float16.NaN > Float16.MinValue);
            Assert.False(Float16.MaxValue > Float16.NaN);
            Assert.False(Float16.NaN == Float16.MinValue);
            Assert.False(Float16.MaxValue == Float16.NaN);
            Assert.True(Float16.MinValue < Float16.MaxValue);
        }

        [Fact(DisplayName = "TestInfinity")]
        public void TestInfinity()
        {
            Assert.False(Float16.IsInfinity(Float16.MinValue));
            Assert.False(Float16.IsInfinity(Float16.MaxValue));

            Float16 posInfinityFromSingle = (Float16)float.PositiveInfinity;
            Assert.True(Float16.IsPositiveInfinity(posInfinityFromSingle));
            Assert.Equal(Float16.PositiveInfinity, posInfinityFromSingle);
            Assert.False(Float16.IsFinite(posInfinityFromSingle));
            Assert.True(Float16.IsInfinity(posInfinityFromSingle));
            Assert.True(Float16.IsPositiveInfinity(posInfinityFromSingle));
            Assert.False(Float16.IsNegativeInfinity(posInfinityFromSingle));

            Assert.False(Float16.IsPositiveInfinity(Float16.MinValue));
            Assert.False(Float16.IsPositiveInfinity(Float16.MaxValue));


            Assert.Equal(float.PositiveInfinity < 0, Float16.IsNegative(posInfinityFromSingle));

            Float16 negInfinityFromSingle = (Float16)float.NegativeInfinity;
            Assert.True(Float16.IsNegativeInfinity(negInfinityFromSingle));
            Assert.Equal(Float16.NegativeInfinity, negInfinityFromSingle);
            Assert.False(Float16.IsFinite(negInfinityFromSingle));
            Assert.True(Float16.IsInfinity(negInfinityFromSingle));
            Assert.True(Float16.IsNegativeInfinity(negInfinityFromSingle));
            Assert.False(Float16.IsPositiveInfinity(negInfinityFromSingle));

            Assert.False(Float16.IsNegativeInfinity(Float16.MinValue));
            Assert.False(Float16.IsNegativeInfinity(Float16.MaxValue));


            Assert.Equal(float.NegativeInfinity < 0, Float16.IsNegative(negInfinityFromSingle));

        }

        [Fact(DisplayName = "TestNormal")]
        public void TestNormal()
        {
            Float16 fp16FromSingleMaxValue = (Float16)float.MaxValue;
            Assert.False(Float16.IsNormal(fp16FromSingleMaxValue));
            Assert.False(Float16.IsNormal(Float16.PositiveInfinity));
            Assert.True(Float16.IsNormal((Float16)45.6f));
            Assert.False(Float16.IsSubnormal((Float16)45.6f));

            Assert.False(Float16.IsSubnormal(fp16FromSingleMaxValue));
            Assert.False(Float16.IsSubnormal(Float16.PositiveInfinity));
        }

        [Fact(DisplayName = "TestEqual")]
        public void TestEqual()
        {
            // Box it
            object obj_1 = Float16.MaxValue;
            object obj_2 = new Float16(Float16.MaxValue.value);
            Assert.True(obj_1.Equals(obj_2));


            Assert.NotEqual(0, obj_1.GetHashCode());
            Assert.Equal(obj_1.GetHashCode(), obj_2.GetHashCode());
            Assert.True(Float16.NaN.Equals(Float16.NaN));

            Float16 fp16Zero = (Float16)0.0f;
            const ushort ushortZero = 0;
            Float16 fp16FromUshortZero = (Float16)ushortZero;

            Assert.True(fp16Zero.Equals(fp16FromUshortZero));

            // Should have the same hash code constant
            Assert.Equal(fp16Zero.GetHashCode(), fp16FromUshortZero.GetHashCode());
            Assert.Equal(Float16.NaN.GetHashCode(), Float16.NaN.GetHashCode());
        }

        [Fact(DisplayName = "TestCompare")]
        public void TestCompare()
        {
            object objMaxValue = new Float16(Float16.MaxValue.value);
            Assert.Equal(0, Float16.MaxValue.CompareTo(objMaxValue));

            Float16 one = (Float16)1.0f;
            Assert.Equal(-1, Float16.MinValue.CompareTo(one));
            Assert.Equal(1, Float16.MaxValue.CompareTo(one));

            // one is bigger than NaN
            Assert.Equal(-1, Float16.NaN.CompareTo(one));
            // Two NaNs are equal according to CompareTo()
            Assert.Equal(0, Float16.NaN.CompareTo((Float16)float.NaN));
            Assert.Equal(1, one.CompareTo(Float16.NaN));

            // Compare to null
            Assert.Equal(1, one.CompareTo(null));

            // Make sure it throws
            var obj = new object();
            Assert.Throws<ArgumentException>(() => one.CompareTo(obj));
        }

    }
}