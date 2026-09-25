# Third-party model sources

The comparison adapters load modules from pinned upstream repositories listed in
`benchmark_sources.json`. Source snapshots are bundled in `.benchmark_sources/`,
with upstream licenses and attribution retained. Nested Git histories, datasets,
weights, generated outputs, binary extensions, notebooks, and image assets are
excluded. Included source files are unmodified copies of the pinned checkouts.
`sources.lock.json` records the upstream revisions and SHA-256 hashes of included
UTF-8 text files with normalized line endings. The loader verifies these snapshots
without requiring nested Git repositories. Existing local Git checkouts continue
to use revision verification. `python run.py --fetch-sources` is optional: it
fetches missing repositories and verifies existing sources without resetting them.

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

Each upstream project's own license and attribution requirements apply. PatchTST
uses Apache-2.0; the other listed repositories include MIT licenses. Nested
third-party notices and licenses are retained where present and continue to apply.
No license is inferred for GaitReader itself or for clinical datasets.
