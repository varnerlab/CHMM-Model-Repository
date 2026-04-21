# Plan: Incorporating Six External References into Paper_v6

Cross-check of the six supplied references against `Paper_v6.tex`, `sections/*_v6.tex`, and `References_v6.bib`.
For each reference, this document states (1) whether it is already cited, (2) where in the paper it naturally belongs, and (3) a concrete insertion-level suggestion.

---

## Executive Summary

| # | Reference | Already in v6? | Action |
|---|---|---|---|
| 1 | Abanto-Valle, Langrock, Chen & Cardoso (2017) | NO | ADD to Related Work + Methods (CHMM-t motivation) |
| 2 | Ang & Bekaert (1998 SSRN; 2002 JBES) | NO | ADD to Related Work (regime-switching paragraph) |
| 3 | Bae, Kim & Mulvey (2014) | NO | ADD to Introduction (motivation) + Conclusion (downstream use) |
| 4 | Hamilton (1989) | YES (`hamilton1989new`) | No action; already anchors regime-switching paragraph |
| 5 | Nguyen (2018) | YES (`nguyen2018hidden`) | No action; already anchors the 4-IC state-count selection |
| 6 | Schaller & Van Norden (1997) | YES (`schaller1997regime`) | No action; already placed alongside Hamilton |

Net: **3 new citations** to add; no existing citations need to move.

---

## 1. Abanto-Valle, Langrock, Chen, Cardoso (2017) -- ASMBI

> *Maximum likelihood estimation for stochastic volatility in mean models with heavy-tailed distributions.* Applied Stochastic Models in Business and Industry 33, 394--408.

**Why it fits.** Directly relevant to the three emission families (CHMM-N / CHMM-t / CHMM-L). The paper fits SV-in-mean models with Normal, Student-t, GED, and Slash error distributions via ML, making it the closest published analog to our decision to ship three emission-family variants of the same EM scaffold. Langrock also has a long track record on HMMs / HSMMs for financial and ecological data, which ties into the Bulla HSMM citation.

**Proposed insertions.**

1. *Related Work, "Stylized facts and parametric volatility" paragraph* (`sections/related_v6.tex`, after the GARCH / jump-diffusion sentences): add one sentence along the lines of:

   > Maximum-likelihood fitting of heavy-tailed innovations in a latent-volatility framework has been studied for stochastic-volatility-in-mean models by \citet{abantovalle2017svm}, who compare Normal, Student-t, GED, and Slash errors; our CHMM-t and CHMM-L are the HMM analog of that choice.

2. *Methods, CHMM-t paragraph* (`sections/methods_v6.tex`): add `\citep{abantovalle2017svm}` after the Peel / Liu-Rubin citations as a non-HMM precedent for Student-t heavy-tailed latent-state fitting.

**BibTeX entry (new in `References_v6.bib`):**
```bibtex
@article{abantovalle2017svm,
  title={Maximum likelihood estimation for stochastic volatility in mean models with heavy-tailed distributions},
  author={Abanto-Valle, Carlos A. and Langrock, Roland and Chen, Ming-Hui and Cardoso, Marcos V.},
  journal={Applied Stochastic Models in Business and Industry},
  volume={33},
  number={4},
  pages={394--408},
  year={2017}
}
```

---

## 2. Ang & Bekaert (1998 SSRN / 2002 JBES)

> *Regime Switches in Interest Rates.* SSRN 94012 (1998); published as Ang, A. & Bekaert, G. (2002), *Journal of Business and Economic Statistics* 20, 163--182.

**Why it fits.** This is one of the most-cited applications of Markov regime-switching *outside* the Hamilton business-cycle context. It documents regime switches in short-rate dynamics and co-movement across countries. Our Related Work paragraph currently lists Hamilton and Schaller & Van Norden as the economics/finance anchors; Ang & Bekaert belongs in the same paragraph because (i) it extends the regime-switching evidence to fixed income, (ii) it is by far the more widely cited of the three, and (iii) it shows the international/cross-asset breadth of regime-switching evidence, which motivates our cross-asset extension.

**Proposed insertion.**

*Related Work, "Regime-switching and the Ryden et al. limitation" paragraph* (`sections/related_v6.tex`, line 7): extend the Hamilton/Schaller sentence:

Current:
> \citet{hamilton1989new} introduced Markov-switching models to economics, and \citet{schaller1997regime} provided an early demonstration that CRSP monthly equity returns are well-described by a two-state Markov-switching specification...

Replacement:
> \citet{hamilton1989new} introduced Markov-switching models to economics; \citet{schaller1997regime} demonstrated the same for monthly CRSP equity returns; and \citet{angbekaert2002regime} showed that short-rate dynamics in several developed markets also admit a regime-switching representation, establishing regime-switching as a cross-asset-class phenomenon rather than one confined to macro series or equities.

**BibTeX entry (new):** Prefer the published JBES version for a citable record. Cite the SSRN working-paper date (1998) if desired for historical ordering.
```bibtex
@article{angbekaert2002regime,
  title={Regime Switches in Interest Rates},
  author={Ang, Andrew and Bekaert, Geert},
  journal={Journal of Business and Economic Statistics},
  volume={20},
  number={2},
  pages={163--182},
  year={2002}
}
```

---

## 3. Bae, Kim & Mulvey (2014) -- EJOR

> *Dynamic asset allocation for varied financial markets under regime switching framework.* European Journal of Operational Research 234, 450--458.

**Why it fits.** This reference documents a concrete downstream use of regime-switching generators: dynamic multi-regime asset allocation. Our paper motivates synthetic-data generators by their use in "risk management or portfolio backtesting" (intro line 2) but does not cite a specific regime-switching asset-allocation paper. Bae-Kim-Mulvey is the cleanest available citation for that motivation and is a natural companion to our cross-asset / copula extension.

**Proposed insertions.**

1. *Introduction* (`sections/introduction_v6.tex`, line 2): strengthen the motivation sentence.

   Current:
   > Any synthetic-data generator intended for risk management or portfolio backtesting must reproduce all three simultaneously \cite{assefa2020generating, jordon2022synthetic}.

   Replacement:
   > Any synthetic-data generator intended for downstream tasks such as risk measurement, scenario generation, or regime-aware dynamic asset allocation \cite{baekimmulvey2014} must reproduce all three simultaneously \cite{assefa2020generating, jordon2022synthetic}.

2. *Conclusion, future-work list* (`sections/conclusion_v6.tex`): add a short bullet on dynamic allocation as a downstream target, citing Bae et al. This signals to referees that the synthetic-data engine has a named client, not just a benchmarking purpose.

**BibTeX entry (new):**
```bibtex
@article{baekimmulvey2014,
  title={Dynamic asset allocation for varied financial markets under regime switching framework},
  author={Bae, Geum Il and Kim, Woo Chang and Mulvey, John M.},
  journal={European Journal of Operational Research},
  volume={234},
  number={2},
  pages={450--458},
  year={2014}
}
```

---

## 4--6. Already-cited references

### Hamilton (1989) -- cited as `hamilton1989new`
Currently the opening citation of the Related Work regime-switching paragraph and the introduction. **No action needed.** The current usage is accurate (regime-switching foundation) and the citation key is well-formed.

### Nguyen (2018) -- cited as `nguyen2018hidden`
Currently cited twice: (a) Abstract and Introduction, where it anchors the four-IC selection (AIC, BIC, HQC, CAIC); (b) Methods `sec:chmm_def` for the IC formulas; (c) Related Work as the prior HMM stock-trading reference. **No action needed.** The 4-IC framework attribution is already credited.

### Schaller & Van Norden (1997) -- cited as `schaller1997regime`
Currently cited in the Introduction and Related Work as the prior evidence that monthly CRSP equity returns admit a two-state Markov-switching representation. **No action needed.** The citation key matches the author order used here (Schaller first; the supplied reference lists Schaller, H. and Norden, S.V.; our v6 bib uses `{Van Norden}, Simon` to respect the Dutch naming convention).

---

## Implementation Plan (if approved)

Atomic edits, kept small so they can be reviewed independently:

1. **Edit `References_v6.bib`** -- add the three new BibTeX entries (`abantovalle2017svm`, `angbekaert2002regime`, `baekimmulvey2014`) in alphabetical position.
2. **Edit `sections/related_v6.tex`** --
   - add the Abanto-Valle sentence at the end of the first paragraph;
   - extend the Hamilton/Schaller sentence with the Ang & Bekaert clause.
3. **Edit `sections/methods_v6.tex`** -- add `\citep{abantovalle2017svm}` alongside Peel / Liu-Rubin in the CHMM-t paragraph.
4. **Edit `sections/introduction_v6.tex`** -- add `\cite{baekimmulvey2014}` in the motivation sentence.
5. **Edit `sections/conclusion_v6.tex`** -- optionally add one future-work bullet pointing at dynamic asset allocation with `\citep{baekimmulvey2014}`.
6. **Recompile `Paper_v6.tex`** and verify no BibTeX warnings.

No changes to results, figures, or tables are required; all three references are literature-only additions.

---

## What is NOT recommended

- Do **not** replace Nguyen with another source for the 4-IC (AIC/BIC/HQC/CAIC) block. Nguyen is the specific prior HMM-on-stock-prices paper that used the same four criteria; swapping in a more general IC reference would weaken the provenance claim.
- Do **not** move Schaller to a lesser position. It is the oldest of the three equity-focused regime-switching references we cite together and should remain the "equity" anchor in the Hamilton/Schaller/Ang-Bekaert triple.
- Do **not** attempt to re-derive any methodological claim from Abanto-Valle. Their SV-in-mean is a *different* latent-state structure (continuous latent log-volatility, not a finite-state Markov chain); the citation is for the heavy-tailed-innovation motivation only, not for any shared algorithm.
