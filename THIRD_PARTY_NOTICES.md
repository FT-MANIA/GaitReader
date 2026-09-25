# Third-party model sources

The comparison adapters load modules from pinned upstream repositories listed in
`benchmark_sources.json`. The upstream source code is not bundled. Running
`python run.py --fetch-sources` explicitly downloads checkouts into the ignored
`.benchmark_sources/` directory. Existing checkouts are verified, not reset.

| Adapter | Upstream |
| --- | --- |
| TimesNet | https://github.com/thuml/Time-Series-Library |
| PatchTST | https://github.com/yuqinie98/PatchTST |
| iTransformer | https://github.com/thuml/iTransformer |
| TS-TCC | https://github.com/emadeldeen24/TS-TCC |
| TS2Vec | https://github.com/zhihanyue/ts2vec |
| T-Rep | https://github.com/let-it-care/t-rep |
| VQShape | https://github.com/YunshiWen/VQShape |
| HeartLang | https://github.com/PKUDigitalHealth/HeartLang |

Each upstream project's own license and attribution requirements apply. Fetching
a repository is not a license grant. Before redistributing upstream code or
weights, review the LICENSE/NOTICE at the pinned revision and retain required
notices. No license is inferred for this project or for clinical datasets.
