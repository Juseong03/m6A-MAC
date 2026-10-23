# paper_data

Data supporting the manuscript "m6A-MAC: target-centered multiscale modeling for RNA m6A site prediction".

- `external_cohorts/`: external candidate tables (HeLa eTAM-seq; GLORI HeLa, HEK293T and mouse MEF).
- `predictions/`: m6A-MAC scores for each of five independently trained runs (seeds 17, 29, 43, 61, 79): internal
  41-801 nt (test and validation), eTAM-seq, SAC-seq (with measured fractions) and GLORI.
- `results/`: per-run metrics and paired contrasts behind the reported tables, including the rows of the literature
  predictors and matched-encoder controls (results only; no comparator code).
- `figure_source/`: values plotted in Figures 2-5.

The internal training, validation and development partitions are in `../data/` (identical to the files used in the
study). Database negatives and uncalled sites are not verified unmethylated sites; SAC-seq and GLORI tables contain
called sites only. These files support the reported results; they do not by themselves reproduce model training.
Checksums: `MANIFEST_SHA256.txt`. Licences and source terms: `LICENSE`, `../DATA_SOURCES.md`.
