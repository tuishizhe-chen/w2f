# report/

The project write-up.

- **`W2F_Final_Report.pdf`** — the compiled report (read this).
- `w2f_report.tex`, `references.bib`, `neurips_2025.sty`, `figs/` — the LaTeX
  source, self-contained for recompilation.

## Rebuild

```bash
pdflatex w2f_report && bibtex w2f_report && pdflatex w2f_report && pdflatex w2f_report
```

(NeurIPS 2025 style; needs a TeX distribution with `natbib`/`plainnat`.)
