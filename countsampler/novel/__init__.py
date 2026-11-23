"""
★ NOVEL ★ The research surface of the project. Everything we tune or change to get results is here.
Everything outside `novel/` (generator wrapper, data, training loop, evaluation) is plumbing.

    novel/
    ├── shared.py              §2   sampler Δφ (Eq. 2) + geometry losses (Eq. 3-9), one file
    ├── losses/
    │   ├── spectral.py        §3   Method A: spectral soft count + gap      (Eq. 10-14)
    │   ├── density.py         §4   Method B: density-map counter loss       (Eq. 16)
    │   └── peak.py            §5   Method C: soft-centre peak separation    (Eq. 18-21)
    ├── components/
    │   └── counter.py         §4.1 C_ψ density counter (Method B)
    └── methods/
        ├── base.py                 objective assembly, stage schedules, step averaging
        ├── method_a.py        §3.4 Eq. 15
        ├── method_b.py        §4.3 Eq. 17
        └── method_c.py        §5   Eq. 22
"""
