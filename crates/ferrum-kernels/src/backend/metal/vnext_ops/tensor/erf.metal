/* origin: FreeBSD /usr/src/lib/msun/src/s_erff.c */
/*
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

// Adapted from Rust libm 0.2.16 / FreeBSD s_erff.c to Metal.
// Only erf is needed; Metal exp follows its separately declared accuracy.
constant float ERX = 8.4506291151e-01f; /* 0x3f58560b */
/*
 * Coefficients for approximation to  erf on [0,0.84375]
 */
constant float EFX8 = 1.0270333290e+00f; /* 0x3f8375d4 */
constant float PP0 = 1.2837916613e-01f; /* 0x3e0375d4 */
constant float PP1 = -3.2504209876e-01f; /* 0xbea66beb */
constant float PP2 = -2.8481749818e-02f; /* 0xbce9528f */
constant float PP3 = -5.7702702470e-03f; /* 0xbbbd1489 */
constant float PP4 = -2.3763017452e-05f; /* 0xb7c756b1 */
constant float QQ1 = 3.9791721106e-01f; /* 0x3ecbbbce */
constant float QQ2 = 6.5022252500e-02f; /* 0x3d852a63 */
constant float QQ3 = 5.0813062117e-03f; /* 0x3ba68116 */
constant float QQ4 = 1.3249473704e-04f; /* 0x390aee49 */
constant float QQ5 = -3.9602282413e-06f; /* 0xb684e21a */
/*
 * Coefficients for approximation to  erf  in [0.84375,1.25]
 */
constant float PA0 = -2.3621185683e-03f; /* 0xbb1acdc6 */
constant float PA1 = 4.1485610604e-01f; /* 0x3ed46805 */
constant float PA2 = -3.7220788002e-01f; /* 0xbebe9208 */
constant float PA3 = 3.1834661961e-01f; /* 0x3ea2fe54 */
constant float PA4 = -1.1089469492e-01f; /* 0xbde31cc2 */
constant float PA5 = 3.5478305072e-02f; /* 0x3d1151b3 */
constant float PA6 = -2.1663755178e-03f; /* 0xbb0df9c0 */
constant float QA1 = 1.0642088205e-01f; /* 0x3dd9f331 */
constant float QA2 = 5.4039794207e-01f; /* 0x3f0a5785 */
constant float QA3 = 7.1828655899e-02f; /* 0x3d931ae7 */
constant float QA4 = 1.2617121637e-01f; /* 0x3e013307 */
constant float QA5 = 1.3637083583e-02f; /* 0x3c5f6e13 */
constant float QA6 = 1.1984500103e-02f; /* 0x3c445aa3 */
/*
 * Coefficients for approximation to  erfc in [1.25,1/0.35]
 */
constant float RA0 = -9.8649440333e-03f; /* 0xbc21a093 */
constant float RA1 = -6.9385856390e-01f; /* 0xbf31a0b7 */
constant float RA2 = -1.0558626175e+01f; /* 0xc128f022 */
constant float RA3 = -6.2375331879e+01f; /* 0xc2798057 */
constant float RA4 = -1.6239666748e+02f; /* 0xc322658c */
constant float RA5 = -1.8460508728e+02f; /* 0xc3389ae7 */
constant float RA6 = -8.1287437439e+01f; /* 0xc2a2932b */
constant float RA7 = -9.8143291473e+00f; /* 0xc11d077e */
constant float SA1 = 1.9651271820e+01f; /* 0x419d35ce */
constant float SA2 = 1.3765776062e+02f; /* 0x4309a863 */
constant float SA3 = 4.3456588745e+02f; /* 0x43d9486f */
constant float SA4 = 6.4538726807e+02f; /* 0x442158c9 */
constant float SA5 = 4.2900814819e+02f; /* 0x43d6810b */
constant float SA6 = 1.0863500214e+02f; /* 0x42d9451f */
constant float SA7 = 6.5702495575e+00f; /* 0x40d23f7c */
constant float SA8 = -6.0424413532e-02f; /* 0xbd777f97 */
/*
 * Coefficients for approximation to  erfc in [1/.35,28]
 */
constant float RB0 = -9.8649431020e-03f; /* 0xbc21a092 */
constant float RB1 = -7.9928326607e-01f; /* 0xbf4c9dd4 */
constant float RB2 = -1.7757955551e+01f; /* 0xc18e104b */
constant float RB3 = -1.6063638306e+02f; /* 0xc320a2ea */
constant float RB4 = -6.3756646729e+02f; /* 0xc41f6441 */
constant float RB5 = -1.0250950928e+03f; /* 0xc480230b */
constant float RB6 = -4.8351919556e+02f; /* 0xc3f1c275 */
constant float SB1 = 3.0338060379e+01f; /* 0x41f2b459 */
constant float SB2 = 3.2579251099e+02f; /* 0x43a2e571 */
constant float SB3 = 1.5367296143e+03f; /* 0x44c01759 */
constant float SB4 = 3.1998581543e+03f; /* 0x4547fdbb */
constant float SB5 = 2.5530502930e+03f; /* 0x451f90ce */
constant float SB6 = 4.7452853394e+02f; /* 0x43ed43a7 */
constant float SB7 = -2.2440952301e+01f; /* 0xc1b38712 */


inline float ferrum_erfc_f32(uint ix, float x) {
    x = abs(x);
    if (ix < 0x3fa00000u) {
        float s = x - 1.0f;
        float p = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
        float q = 1.0f + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));
        return (1.0f - ERX) - p / q;
    }
    float s = 1.0f / (x * x), r, t;
    if (ix < 0x4036db6du) {
        r = RA0 + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7))))));
        t = 1.0f + s * (SA1 + s * (SA2 + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8)))))));
    } else {
        r = RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6)))));
        t = 1.0f + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7))))));
    }
    float z = as_type<float>(as_type<uint>(x) & 0xffffe000u);
    return exp(-z * z - 0.5625f) * exp((z - x) * (z + x) + r / t) / x;
}
inline float ferrum_erf_f32(float x) {
    uint bits = as_type<uint>(x), ix = bits & 0x7fffffffu;
    if (ix >= 0x7f800000u) return 1.0f - 2.0f * float(bits >> 31) + 1.0f / x;
    if (ix < 0x3f580000u) {
        if (ix < 0x31800000u) return 0.125f * (8.0f * x + EFX8 * x);
        float z = x * x;
        float r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
        float s = 1.0f + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
        return x + x * (r / s);
    }
    float y = ix < 0x40c00000u ? 1.0f - ferrum_erfc_f32(ix, x) : 1.0f;
    return (bits >> 31) != 0 ? -y : y;
}
