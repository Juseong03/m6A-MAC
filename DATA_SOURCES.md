# Data sources and terms

| Files | Derived from | Terms |
|---|---|---|
| `data/*.tsv.gz` (internal HEK293T partitions) | m6AConquer HEK293T table (Zhao et al., Nucleic Acids Res. 2026, doi:10.1093/nar/gkaf1204); RefSeq annotation via UCSC | m6AConquer is freely available without login; its website states "All Rights Reserved" and gives no explicit redistribution licence. Cite m6AConquer. |
| `paper_data/external_cohorts/eTAM-seq_HeLa/` | GEO GSE211303 (eTAM-seq) | GEO: NCBI places no restrictions on use or redistribution; submitters may hold rights. Cite the original study. |
| `paper_data/predictions/SAC-seq_HeLa/` (site fractions) | GEO GSE198246 (m6A-SAC-seq) | As above. |
| `paper_data/external_cohorts/GLORI/` | GEO GSE210563, GSE270643 (GLORI, GLORI 2.0) | As above. |
| All other `paper_data/` files | Created in this study | CC BY 4.0 (paper_data/LICENSE) |

The code (src/, run.py, tests/) is under the MIT licence (LICENSE). RiNALMo is a separate dependency under Apache-2.0
(licenses/RiNALMo.LICENSE); its weights are not included. No comparator-model code is included.
