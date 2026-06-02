# Fix: Portfolio CSV Not Loading

## Root Cause

**Primary bug — comma inside inline comments:**

Row 4 of [`portfolio.csv`](../portfolio.csv:4):
```
ITA,7.99,99.93          #Global Aerospace, Defence
```

`pd.read_csv()` does **not** treat `#` as a comment character by default. Since `#Global Aerospace, Defence` contains a comma, pandas sees **4 columns** on this row, while the header has only 3. This raises a `ParserError`, caught by the bare `except Exception` in [`load_portfolio()`](../portfolio.py:26-27), which returns an **empty DataFrame**.

**Secondary issue — trailing tabs + comments cause NaN:**

Even on rows without an extra comma (e.g. `IWDA.AS,121.19,77.47\t\t\t#Core MSCI World USD`), the trailing tab characters and `#...` text are included in the field value. `pd.to_numeric("77.47\t\t\t#Core MSCI World USD", errors="coerce")` returns `NaN`, wiping out `Buy_Price` and `Amount_EUR`.

## Fix (2-line change in `portfolio.py`)

### Change 1 — Add `comment="#"` parameter

```python
df = pd.read_csv(filepath, comment="#").dropna(how="all")
```

- Tells pandas to strip everything from `#` to end-of-line **before** CSV field parsing
- Eliminates the stray comma problem entirely
- Also removes trailing tab/comment artifacts

### Change 2 — Strip whitespace before numeric conversion

```python
df["Buy_Price"] = pd.to_numeric(df["Buy_Price"].astype(str).str.strip(), errors="coerce")
df["Amount_EUR"] = pd.to_numeric(df["Amount_EUR"].astype(str).str.strip(), errors="coerce")
```

- `comment="#"` strips the `#...` but leaves trailing whitespace/tabs
- `.str.strip()` cleans the value before `pd.to_numeric()` can reject it

## Full Diff of `portfolio.py`

```diff
     try:
-        df = pd.read_csv(filepath).dropna(how="all")
+        df = pd.read_csv(filepath, comment="#").dropna(how="all")
         df.columns = df.columns.str.strip()
 
         for col in required_cols:
             if col not in df.columns:
                 df[col] = 0.0 if col != "Symbol" else "UNKNOWN"
 
         df["Symbol"] = df["Symbol"].astype(str).str.strip()
-        df["Buy_Price"] = pd.to_numeric(df["Buy_Price"], errors="coerce")
-        df["Amount_EUR"] = pd.to_numeric(df["Amount_EUR"], errors="coerce")
+        df["Buy_Price"] = pd.to_numeric(df["Buy_Price"].astype(str).str.strip(), errors="coerce")
+        df["Amount_EUR"] = pd.to_numeric(df["Amount_EUR"].astype(str).str.strip(), errors="coerce")
         return df.dropna(subset=["Symbol"]).reset_index(drop=True)
```

## Verification

After the fix, run `python3 main.py`. The output should show:

```
Loaded Portfolio: ['IWDA.AS', 'PHO', 'ITA', 'HMY', 'LMT', 'ROG.SW']
```

And the portfolio audit section should appear at the end of the scan with real PnL calculations.

```mermaid
flowchart LR
    CSV["portfolio.csv\nwith # comments"] --> PARSE["pd.read_csv\ncomment=#"]
    PARSE --> CLEAN["Strip whitespace\n.str.strip()"]
    CLEAN --> NUM["pd.to_numeric\nerrors=coerce"]
    NUM --> DF["Portfolio DataFrame\nSymbols with valid prices"]
    DF --> AUDIT["audit_portfolio()\nPnL & decisions"]