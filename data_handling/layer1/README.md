# SahayakGraph - Layer 1 (Data) - Gemini version

Seven step scripts plus a Gemini connection test. Run them in order.
Each step reads the previous step's output. Your input CSV is only ever read.

## Folder layout

```
D:\Sahayak_Graph\
  .env                      <- GEMINI_API_KEY=...   (saved as UTF-8)
  data\raw\schemes_clean.csv   <- YOUR input (never modified)
  data\interim\             <- work in progress, cached Gemini answers, logs
  data\processed\           <- final Layer 1 outputs
  layer1\ (or the root)     <- the scripts
```
The scripts find the project root automatically (the folder holding `.env` and `data\`).

## Creating the .env file correctly (important on Windows)

Do NOT use `echo "GEMINI_API_KEY=..." > .env` in Windows PowerShell: it saves UTF-16 and the key
cannot be read. Use Notepad (Save as, Encoding: UTF-8) or:

    Set-Content -Path .env -Value "GEMINI_API_KEY=paste-your-key-here" -Encoding utf8

## Run order

| Step | Command | Output |
|---|---|---|
| test | `python test_gemini.py` | PASS/FAIL for key, connection, JSON output, cut-off detection |
| A | `python step_a_clean.py` | `interim\schemes_step_a.csv` |
| B | `python step_b_vocabulary.py` | `processed\attribute_vocabulary.json` |
| C | `python step_c_extract_conditions.py --pilot`, then `--all` | `processed\conditions.csv` |
| D | `python step_d_spot_check.py checks` / `sheet` / `score` | `processed\spot_check_sheet.csv`, `spot_check_report.txt` |
| E | `python step_e_documents.py` | `processed\documents.csv` |
| F | `python step_f_application_department.py` | `processed\schemes_clean.csv`, `application_steps.csv` |
| G | `python step_g_conflicts.py candidates`, edit `keep`, then `finalize` | `processed\conflicts.csv` |
| - | `python check_layer1.py` | PASS/FAIL completeness list |

## How Gemini answers are handled

- Every request asks for JSON that follows a schema (structured output).
- A reply is accepted only if Gemini finished normally, the text is non-empty, valid JSON,
  and matches the schema. Otherwise it is retried (a cut-off reply is retried with double the
  token budget). Nothing is repaired or guessed.
- Rate limits and server errors (429, 5xx) are retried with waiting. A wrong key or a
  wrong/retired model name stops the run immediately with a clear message.
- A scheme that still fails is skipped and listed in `data\interim\llm_failures.log`.
  Re-running the step retries only those schemes (good answers are cached).
- Settings (model, retries, token limits) are in `.env`; see `.env.example`.
