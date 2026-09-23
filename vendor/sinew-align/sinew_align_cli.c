/*
 * sinew_align_cli -- the org's rotation fitter (sinew_align.c, Align.lean's
 * port) as a host command, so analysis scripts fit rotations through it
 * instead of their own code (gates/6d-fit-avbd/ladder_eval.py).
 *
 *   sinew_align_cli PAIRS.txt [--about-y]
 *
 * PAIRS.txt: one pair per line, "tx ty tz sx sy sz" (target, source), any
 * count; the points are centred here (each set on its own centroid) and
 * sinew_align recovers R with target - ct ~= R (source - cs). Prints the
 * row-major 3x3 on one line, then the two centroids. --about-y restricts
 * the fit to a rotation about y: the centred pairs keep only x and z (y = 0)
 * and the y axis itself goes in as weighted pairs (see main), so the fitter
 * returns the rotation about y.
 * SPDX-License-Identifier: MIT
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sinew_align.h"

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: sinew_align_cli PAIRS.txt [--about-y]\n");
        return 2;
    }
    int about_y = argc > 2 && strcmp(argv[2], "--about-y") == 0;
    FILE *f = fopen(argv[1], "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    size_t cap = 1024, n = 0;
    double *t = malloc(cap * 3 * sizeof(double)), *s = malloc(cap * 3 * sizeof(double));
    double row[6];
    while (fscanf(f, "%lf %lf %lf %lf %lf %lf", &row[0], &row[1], &row[2], &row[3], &row[4], &row[5]) == 6) {
        if (n == cap) {
            cap *= 2;
            t = realloc(t, cap * 3 * sizeof(double));
            s = realloc(s, cap * 3 * sizeof(double));
        }
        for (int k = 0; k < 3; k++) {
            t[3 * n + k] = row[k];
            s[3 * n + k] = row[3 + k];
        }
        n++;
    }
    fclose(f);
    if (n == 0) {
        fprintf(stderr, "no pairs\n");
        return 2;
    }
    double ct[3] = { 0, 0, 0 }, cs[3] = { 0, 0, 0 };
    for (size_t i = 0; i < n; i++) {
        for (int k = 0; k < 3; k++) {
            ct[k] += t[3 * i + k];
            cs[k] += s[3 * i + k];
        }
    }
    for (int k = 0; k < 3; k++) {
        ct[k] /= (double)n;
        cs[k] /= (double)n;
    }
    double radius = 0.0;
    for (size_t i = 0; i < n; i++) {
        for (int k = 0; k < 3; k++) {
            t[3 * i + k] -= ct[k];
            s[3 * i + k] -= cs[k];
        }
        if (about_y) {
            t[3 * i + 1] = 0.0;
            s[3 * i + 1] = 0.0;
        }
        for (int k = 0; k < 3; k++) {
            double v = fabs(s[3 * i + k]);
            if (v > radius) radius = v;
        }
    }
    size_t m = n;
    if (about_y) {
        /* The y axis as pairs (0, +-L, 0) -> (0, +-L, 0), L twice the data's
         * radius, n/4 times over: a rotation that moved the axis (the
         * 180-degree flip about an in-plane axis that coplanar data alone
         * leaves free; it took one height band of Gate 6d's sweep) costs
         * 8 n r^2, more than any in-plane misfit (at most 4 n r^2), and the
         * covariance stays as well conditioned as the data (an L of a
         * thousand radii swamped it: ns30 left the in-plane block unresolved
         * and the identity came back). */
        const double L = 2.0 * (radius > 0.0 ? radius : 1.0);
        const size_t reps = n / 4 > 1 ? n / 4 : 1;
        if (n + 2 * reps > cap) {
            cap = n + 2 * reps;
            t = realloc(t, cap * 3 * sizeof(double));
            s = realloc(s, cap * 3 * sizeof(double));
        }
        for (size_t rep = 0; rep < reps; rep++) {
            for (int sign = -1; sign <= 1; sign += 2) {
                t[3 * m] = 0.0; t[3 * m + 1] = sign * L; t[3 * m + 2] = 0.0;
                s[3 * m] = 0.0; s[3 * m + 1] = sign * L; s[3 * m + 2] = 0.0;
                m++;
            }
        }
    }
    double r[9];
    sinew_align(t, s, m, r);
    printf("R %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n", r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]);
    printf("ct %.10f %.10f %.10f\n", ct[0], ct[1], ct[2]);
    printf("cs %.10f %.10f %.10f\n", cs[0], cs[1], cs[2]);
    printf("valid9 %d n %zu\n", sinew_valid9(r), n);
    free(t);
    free(s);
    return 0;
}
