---
paths:
  - "**/*.tex"
  - "**/*.bib"
---

# LaTeX Paper Conventions

> Writing standards for the Para_space manuscript.
> Target venues: NeurIPS / ICML / CVPR / TPAMI.

---

## 1. File Organization

```
paper/
  main.tex              ← Orchestrating file (preamble, \input chain)
  sections/
    00_abstract.tex     ← Numbered for ordering
    01_introduction.tex
    02_related.tex
    ...
    A_theory_proofs.tex ← Appendix prefixed with letter
  refs/
    main.bib            ← All references
  figures/              ← All figures (PDF or PNG, 300+ dpi)
```

**Rules**:
- One section = one `.tex` file
- Number prefix determines ordering in `main.tex`
- Appendix files use letter prefix (`A_`, `B_`, `C_`)
- Figures live in `figures/`, referenced by relative path

---

## 2. Citation Style

- Use `\citep{}` for parenthetical citations, `\citet{}` for in-text
- Bibliography: `\bibliographystyle{plainnat}` (author-year)
- Key references to always cite: SIREN (Sitzmann 2020), LIIF (Chen 2021), COIN (Dupont 2022), MAML (Finn 2017), NTK (Jacot 2018)
- Before adding a new reference, check `refs/main.bib` — it may already be there

---

## 3. Math Notation Conventions

| Concept | Notation | Command |
|---------|----------|---------|
| Parameter space | $\Theta$ | `\Theta` |
| Optimal parameters | $\theta^*$ | `\theta^*` |
| Lie algebra | $\mathfrak{g}$ | `\g` (defined in preamble) |
| Rotation group | $\text{SO}(2)$ | `\SO` (defined in preamble) |
| Real numbers | $\mathbb{R}$ | `\R` (defined in preamble) |
| Tangent vector | $J_X(f)$ | `J_X(f)` |
| Symmetry error | $\epsilon_{\text{sym}}$ | `\eps_{\text{sym}}` |

**Rules**:
- Use preamble-defined commands, not raw `\mathbb{}` or `\mathcal{}` — keeps notation consistent
- New commands go in `main.tex` preamble, not in individual section files
- Vector notation: `\vnat{}` for bold vectors, `\mathbf{}` for matrices

---

## 4. Figure and Table Standards

- Every figure must have a `\caption{}` and `\label{fig:...}`
- Every table must have a `\caption{}` and `\label{tab:...}`
- Tables use `\begin{tabular}` with `\toprule`, `\midrule`, `\bottomrule` (booktabs)
- Figures: vector format preferred (PDF), PNG at ≥ 300 dpi as fallback
- Reference figures with `Fig.~\ref{fig:...}`, tables with `Table~\ref{tab:...}`

---

## 5. Writing Style

- **Active voice** in abstract and intro; passive acceptable in methods
- Each section should answer exactly one question:
  - Introduction: *Why does this matter?*
  - Related Work: *What's been done and what's missing?*
  - Formulation: *What exactly are we studying?*
  - Theory: *What does the math predict?*
  - Experiments: *Does reality match the prediction?*
- **Limitations** section is mandatory — list at least 3 concrete boundaries
- Every claim in the abstract must be supported by a result in the experiments section

---

## 6. Build and Compilation

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex   # Third pass for cross-refs
```

Use `latexmk -pdf main.tex` for automatic recompilation during editing.

---

## 7. Collaboration

- Each section file is self-contained — can be edited independently
- Mark unfinished parts with `\todo{}` from the `todonotes` package
- Track changes: comment old text with `%` rather than deleting (for review)
- Before sharing the PDF, remove or hide all `\todo{}` notes
