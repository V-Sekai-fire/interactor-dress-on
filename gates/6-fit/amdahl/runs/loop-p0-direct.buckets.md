| bucket | s | fraction | ms / Newton (100) | of which alloc/free |
|---|---|---|---|---|
| gradient | 0.29 | 0.9% | 2.9 | 2% |
| hessian: ContactForm | 0.18 | 0.6% | 1.8 | 18% |
| hessian: CurveCurvatureForm | 0.07 | 0.2% | 0.7 | 7% |
| hessian: CurveTargetForm | 0.03 | 0.1% | 0.3 | 17% |
| hessian: CurveTorsionForm | 0.35 | 1.1% | 3.5 | 5% |
| hessian: PointPenaltyForm | 0.06 | 0.2% | 0.6 | 18% |
| hessian: SimilarityForm | 3.11 | 10.0% | 31.1 | 11% |
| hessian: sum of forms, P^T H P, copies | 1.89 | 6.1% | 18.9 | 17% |
| linear: analyze | 0.73 | 2.3% | 7.3 | 11% |
| linear: factorize | 2.69 | 8.7% | 26.9 | 0% |
| linear: other (H*dx residual) | 0.06 | 0.2% | 0.6 | 55% |
| linear: solve | 0.09 | 0.3% | 0.9 | 0% |
| broad phase: CCD candidates (x0->x1) | 13.28 | 42.7% | 132.8 | 13% |
| broad phase: constraint set (dhat) | 0.16 | 0.5% | 1.6 | 14% |
| CCD / max step size | 6.50 | 20.9% | 65.0 | 18% |
| line search: energy evals | 0.50 | 1.6% | 5.0 | 22% |
| line search: other | 0.08 | 0.3% | 0.8 | 21% |
| constraint-set update | 0.70 | 2.3% | 7.0 | 1% |
| energy (Newton loop) | 0.20 | 0.6% | 2.0 | 18% |
| other (Newton loop bookkeeping) | 0.05 | 0.2% | 0.5 | 4% |
| other (phase driver, AL outer loop) | 0.09 | 0.3% | 0.9 | 50% |
| **phase window (FitDriver::step)** | **31.10** | 100% | 311 | 13% |
| outside the window (inputs, begin, exit) | 0.59 | - | - | - |

| group | s | fraction |
|---|---|---|
| gradient | 0.29 | 0.9% |
| hessian | 5.69 | 18.3% |
| linear | 3.57 | 11.5% |
| broad phase | 13.44 | 43.2% |
| CCD / max step size | 6.50 | 20.9% |
| line search | 0.58 | 1.9% |
| constraint-set update | 0.70 | 2.3% |
| energy (Newton loop) | 0.20 | 0.6% |
| other (Newton loop bookkeeping) | 0.05 | 0.2% |
| other (phase driver, AL outer loop) | 0.09 | 0.3% |
| allocation overlay (stacks with an allocator frame, across buckets) | 3.95 | 12.7% |
