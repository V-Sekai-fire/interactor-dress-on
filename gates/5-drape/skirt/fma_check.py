"""The bending hinge G10 found (bending 2443 of the fitted skirt at drape scale
10): its s = sum w_i p_i per component in float32, summed as the kernel is
written (seq), and with the two FMA contractions a GPU compiler may form
(fma a: fma(w1,x1,w0*x0) + fma(w3,x3,w2*x2); fma b: the other pairing),
against the exact sum. Inputs are the rest positions and weights bisect
printed."""
import numpy as np, itertools
f = np.float32
w = [f(2.25989437), f(5.04274893), f(-3.71447849), f(-3.58816504)]
P = [(f(2.14215755), f(6.84740829), f(-0.731058121)), (f(2.03971624), f(7.2909503), f(-0.87708056)),
     (f(2.06092691), f(7.37199402), f(-0.772039831)), (f(2.08227825), f(6.92770243), f(-0.89385128))]

def fma(a, b, c):
    return f(float(a) * float(b) + float(c))

for comp in range(3):
    x = [P[i][comp] for i in range(4)]
    seq = f(f(f(w[0] * x[0]) + f(w[1] * x[1])) + f(f(w[2] * x[2]) + f(w[3] * x[3])))
    fm1 = f(fma(w[1], x[1], f(w[0] * x[0])) + fma(w[3], x[3], f(w[2] * x[2])))
    fm2 = f(fma(w[0], x[0], f(w[1] * x[1])) + fma(w[2], x[2], f(w[3] * x[3])))
    exact = sum(float(w[i]) * float(x[i]) for i in range(4))
    print("xyz"[comp], "seq", seq, "fma a", fm1, "fma b", fm2, "exact", exact)
