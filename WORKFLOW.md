# CreditCast: One-Day Workflow (fast path)

Deadline: today. Scope = guide's **fast path**: Phases 0-7, quick Phase 8, short README, then the resume harvest.
Cut for today (mark "NOT BUILT" so they stay off the resume): Streamlit demo, CI, full SR 11-7 PDF, prepayment model (unless time remains).

Rule: every resume number comes from `results/*.json`. After each phase, run the "explain + quiz me" prompt.

## Time budget (about 12 working hours)

| Block | Phase | Time | Owner | Output |
|---|---|---|---|---|
| A | 0. Data + keys | 60-90 min | **You** | Freddie files + user guide PDF, FRED key, Fed PDF |
| B | Scaffold | 15 min | Claude | CLAUDE.md, config.yaml, .gitignore, Makefile skeleton |
| C | 1. Panel | 90 min | Claude + you check | `data_summary.json`, crisis spike plot |
| D | 2. Macro, Fed, ARIMA | 60 min | Claude + you verify Fed numbers | `arima.json`, `fed_2026_scenarios.csv` |
| E | 3. Features + K-Means | 45 min | Claude | `segments.json` |
| F | 4. PD model | 60 min | Claude | `pd_metrics.json` (train/val only) |
| G | 5. LGD + EAD | 45 min | Claude | `lgd.json` |
| H | 6. ECL engine | 60 min | Claude | `ecl.json` |
| I | 7. Stress test | 45 min | Claude | `stress.json`, heatmap |
| J | 8. Validation (quick) | 60 min | Claude | `backtest.json`, `psi.json` |
| K | 10-lite + 11 | 60 min | Claude | README, tests green, `RESUME_NUMBERS.md` |

Bring a ready-to-send `RESUME_NUMBERS.md` by the end. If you are behind at block H, skip prepayment and document the limitation.

## Step-by-step

### Step 0: You, now (start the downloads first, they are the long pole)
1. Register at Freddie Mac (Clarity). Download sample files: about 3 pre-crisis years (2004-2008) and 3-4 recent years (2012-2019), plus the **user guide PDF**. Confirm loan counts per year.
2. Get a FRED API key.
3. Download the Fed 2026 scenarios PDF.
4. Put the files in `data/raw/` and the PDFs in `docs/reference/`. Put the key in `.env` as `FRED_API_KEY=...`.
5. Install Python 3.11+: `python3 -m venv .venv && source .venv/bin/activate`.

### Step 1: Scaffold (Claude, done in this session)
`CLAUDE.md`, `config.yaml`, `.gitignore`, `WORKFLOW.md`. Next: `pyproject.toml`, package folders, Makefile, empty test stubs.

### Steps 2-9: Run Phases 1-8 from the guide, in order
For each phase:
1. Start Claude Code in plan mode, paste that phase's prompt from the guide, read the plan, say "go".
2. Check the "Done when" item (see the gates below).
3. Run `pytest`; it must be green.
4. Ask: "Explain what you built as if I'm in an interview, then quiz me with 5 questions."
5. Commit using the message I give you.

### Gates (don't move on until true)
- P1: 2008-2012 spike visible; training defaults >= ~500; removed-row counts documented.
- P2: Fed numbers match the PDF by eye (10% unemployment, ~30% HPI drop, 4.6% GDP drop).
- P3: k chosen by elbow + silhouette on training data only; VIF <= 10.
- P4: expected coefficient signs pass; OOT test not touched yet.
- P5: LGD rises when HPI falls; amortisation reaches zero.
- P6: survival in [0,1] and non-increasing; severe >= baseline.
- P7: heatmap monotone in both axes.
- P8: first and only use of the OOT set; PSI table; COVID appendix.

### Step 10: Wrap-up
1. Phase 10-lite: Makefile targets, README (results table from JSON, limitations), `pytest` green, `git ls-files | grep data/raw` empty.
2. Phase 11: `make resume_numbers` -> `results/RESUME_NUMBERS.md`.
3. Send me that file's contents.

## Risks today
- Freddie registration/download delay: the biggest risk. Do Step 0 first.
- Too few defaults: add pre-crisis vintages.
- Fed PDF table extraction may need manual fixing: verify by eye.
- Time overrun: drop in this order: prepayment, sensitivity grid size, ARIMA breadth, XGBoost tuning.
