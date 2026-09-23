| bucket | s | fraction | ms / Newton (13) | of which alloc/free |
|---|---|---|---|---|
| gradient | 0.09 | 0.8% | 6.6 | 5% |
| hessian: ContactForm | 0.04 | 0.3% | 2.8 | 16% |
| hessian: CurveCurvatureForm | 0.03 | 0.2% | 2.1 | 15% |
| hessian: CurveTargetForm | 0.01 | 0.1% | 0.9 | 33% |
| hessian: CurveTorsionForm | 0.09 | 0.8% | 6.9 | 6% |
| hessian: PointPenaltyForm | 0.01 | 0.1% | 1.0 | 15% |
| hessian: SimilarityForm | 3.24 | 29.0% | 249.4 | 8% |
| hessian: sum of forms, P^T H P, copies | 1.30 | 11.6% | 99.7 | 19% |
| linear: factorize | 0.01 | 0.1% | 0.9 | 0% |
| linear: other (H*dx residual) | 0.01 | 0.1% | 1.1 | 71% |
| linear: solve | 2.51 | 22.5% | 192.9 | 0% |
| broad phase: CCD candidates (x0->x1) | 3.21 | 28.8% | 247.3 | 9% |
| broad phase: constraint set (dhat) | 0.21 | 1.9% | 16.2 | 10% |
| CCD / max step size | 0.09 | 0.8% | 6.8 | 11% |
| line search: energy evals | 0.10 | 0.9% | 7.8 | 26% |
| line search: other | 0.01 | 0.1% | 0.8 | 0% |
| constraint-set update | 0.01 | 0.1% | 1.2 | 0% |
| energy (Newton loop) | 0.10 | 0.9% | 7.3 | 21% |
| other (Newton loop bookkeeping) | 0.00 | 0.0% | 0.3 | 0% |
| other (phase driver, AL outer loop) | 0.09 | 0.8% | 7.2 | 47% |
| **phase window (FitDriver::step)** | **11.17** | 100% | 859 | 9% |
| outside the window (inputs, begin, exit) | 0.65 | - | - | - |

| group | s | fraction |
|---|---|---|
| gradient | 0.09 | 0.8% |
| hessian | 4.72 | 42.2% |
| linear | 2.53 | 22.7% |
| broad phase | 3.43 | 30.7% |
| CCD / max step size | 0.09 | 0.8% |
| line search | 0.11 | 1.0% |
| constraint-set update | 0.01 | 0.1% |
| energy (Newton loop) | 0.10 | 0.9% |
| other (Newton loop bookkeeping) | 0.00 | 0.0% |
| other (phase driver, AL outer loop) | 0.09 | 0.8% |
| allocation overlay (stacks with an allocator frame, across buckets) | 0.96 | 8.6% |
