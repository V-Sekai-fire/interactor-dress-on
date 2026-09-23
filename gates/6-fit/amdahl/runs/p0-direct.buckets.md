| bucket | s | fraction | ms / Newton (41) | of which alloc/free |
|---|---|---|---|---|
| gradient | 0.24 | 0.6% | 5.8 | 2% |
| hessian: ContactForm | 0.08 | 0.2% | 2.0 | 17% |
| hessian: CurveCurvatureForm | 0.04 | 0.1% | 1.0 | 19% |
| hessian: CurveTargetForm | 0.01 | 0.0% | 0.3 | 23% |
| hessian: CurveTorsionForm | 0.16 | 0.4% | 3.8 | 4% |
| hessian: PointPenaltyForm | 0.02 | 0.1% | 0.6 | 4% |
| hessian: SimilarityForm | 4.37 | 11.2% | 106.6 | 10% |
| hessian: sum of forms, P^T H P, copies | 2.52 | 6.5% | 61.5 | 19% |
| linear: analyze | 0.91 | 2.3% | 22.2 | 11% |
| linear: factorize | 4.56 | 11.7% | 111.3 | 0% |
| linear: other (H*dx residual) | 0.04 | 0.1% | 0.9 | 47% |
| linear: solve | 0.13 | 0.3% | 3.3 | 1% |
| broad phase: CCD candidates (x0->x1) | 8.92 | 22.9% | 217.7 | 12% |
| broad phase: constraint set (dhat) | 0.22 | 0.6% | 5.4 | 11% |
| CCD / max step size | 13.86 | 35.6% | 337.9 | 21% |
| line search: energy evals | 0.44 | 1.1% | 10.8 | 26% |
| line search: other | 0.04 | 0.1% | 1.0 | 7% |
| constraint-set update | 1.97 | 5.1% | 48.1 | 0% |
| energy (Newton loop) | 0.21 | 0.5% | 5.0 | 20% |
| other (Newton loop bookkeeping) | 0.06 | 0.2% | 1.5 | 2% |
| other (phase driver, AL outer loop) | 0.11 | 0.3% | 2.8 | 49% |
| **phase window (FitDriver::step)** | **38.94** | 100% | 950 | 14% |
| outside the window (inputs, begin, exit) | 0.77 | - | - | - |

| group | s | fraction |
|---|---|---|
| gradient | 0.24 | 0.6% |
| hessian | 7.21 | 18.5% |
| linear | 5.65 | 14.5% |
| broad phase | 9.14 | 23.5% |
| CCD / max step size | 13.86 | 35.6% |
| line search | 0.48 | 1.2% |
| constraint-set update | 1.97 | 5.1% |
| energy (Newton loop) | 0.21 | 0.5% |
| other (Newton loop bookkeeping) | 0.06 | 0.2% |
| other (phase driver, AL outer loop) | 0.11 | 0.3% |
| allocation overlay (stacks with an allocator frame, across buckets) | 5.33 | 13.7% |
