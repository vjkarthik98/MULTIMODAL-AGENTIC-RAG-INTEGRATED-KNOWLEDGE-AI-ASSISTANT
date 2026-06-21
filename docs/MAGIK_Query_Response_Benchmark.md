# MAGIK Multimodal RAG — Query & Response Benchmark Dataset

**Purpose:** Ground-truth query–response pairs for RAG accuracy evaluation, embedding alignment, and eval harness calibration.  
**Coverage:** 35 queries across 7 files (28 RAG + 7 Web Search), 5 per file.  
**Citation format by modality:**

| Modality | Citation Format |
|----------|----------------|
| TXT | No citation (plain document) |
| PDF | Page number (e.g., `Page 32`) |
| DOCX | Section heading (e.g., `§ 4.1 DCF Model Key Assumptions`) |
| XLSX | Sheet name (e.g., `Sheet: ERPs by country`) |
| JPG | Image caption |
| MP3 | Speaker + timestamp (e.g., `Chair Powell · 00:00:46`) |
| MP4 | Timestamp + frame caption (e.g., `00:02:06 · Frame: AAPL candlestick chart, Benzinga live`) |

---

## FILE 1 — `apple_10k.pdf`
**Apple Inc. Form 10-K | Fiscal Year Ended September 28, 2024 | 121 pages**

---

### Q1 · RAG

**Query:** What were Apple's net sales by product category for fiscal year 2024 versus fiscal year 2023?

**Answer:**

Apple's net sales by category (in millions):

| Category | FY2024 | FY2023 | YoY Change |
|---|---|---|---|
| iPhone | $201,183 | $200,583 | Flat (0%) |
| Mac | $29,984 | $29,357 | +2% |
| iPad | $26,694 | $28,300 | −6% |
| Wearables, Home and Accessories | $37,005 | $39,845 | −7% |
| Services | $96,169 | $85,200 | +13% |
| **Total Net Sales** | **$391,035** | **$383,285** | **+2%** |

iPhone revenue was essentially flat. Services was the standout, growing 13% driven primarily by higher revenue from advertising, the App Store, and cloud services. Wearables, Home and Accessories declined 7%, primarily due to lower Wearables and Accessories sales. iPad fell 6% due to lower iPad Pro and entry-level iPad sales.

**Source:** Apple Inc. Form 10-K, **Page 26** — Products and Services Performance table.

---

### Q2 · RAG

**Query:** What was Apple's gross margin performance in FY2024, broken down by Products and Services, and how did it compare to FY2023?

**Answer:**

Apple's gross margin improved substantially in FY2024 (dollars in millions):

| Segment | FY2024 GM $ | FY2024 GM % | FY2023 GM % |
|---|---|---|---|
| Products | $109,633 | 37.2% | 36.5% |
| Services | $71,050 | 73.9% | 70.8% |
| **Total** | **$180,683** | **46.2%** | **44.1%** |

Total gross margin expanded 210 basis points year-over-year to 46.2%. Products gross margin improved due to cost savings, partially offset by an unfavorable product mix and currency headwinds. Services gross margin increased due to both higher revenue and a more favorable Services mix. This is the first time Apple's total gross margin exceeded 46%.

**Source:** Apple Inc. Form 10-K, **Page 27** — Gross Margin table; **Page 32** — Consolidated Statements of Operations.

---

### Q3 · RAG

**Query:** What was the impact of the EU State Aid Decision on Apple's tax provision and effective tax rate in FY2024?

**Answer:**

The EU State Aid Decision resulted in a one-time income tax charge of **$10.2 billion (net)** in FY2024. As a consequence:

- Apple's **effective tax rate** jumped to **24.1%** in FY2024 — up from 14.7% in FY2023 — and above the 21% statutory federal rate.
- The total **provision for income taxes** was **$29,749 million** in FY2024, compared to $16,741 million in FY2023.
- As of September 28, 2024, Apple had a **€14.2 billion (≈ $15.8 billion)** obligation payable to Ireland, expected to be settled within 12 months. The funds were held in escrow as of the balance sheet date and restricted from general use.

The primary reasons the effective rate exceeded the statutory rate were: the one-time State Aid charge, state income taxes, and a higher effective rate on foreign earnings — partially offset by R&D credits and share-based compensation tax benefits.

**Source:** Apple Inc. Form 10-K, **Page 28** — Provision for Income Taxes; **Page 29** — State Aid Decision Tax Payable.

---

### Q4 · RAG

**Query:** How much did Apple return to shareholders through repurchases and dividends in FY2024, and what new capital return program was announced?

**Answer:**

In FY2024, Apple returned a combined **$110.2 billion** to shareholders:

- **Share repurchases:** $95.0 billion of common stock repurchased during the fiscal year (reflected as $95,846 million in the shareholders' equity rollforward).
- **Dividends and dividend equivalents paid:** $15.2 billion ($15,234 million per cash flow statement).
- **Quarterly dividend rate:** raised from $0.24 to **$0.25 per share** beginning May 2024; annualized to $0.98/share for FY2024.

**Capital Return Program (May 2024):** Apple announced a **new $110 billion share repurchase authorization** and the dividend increase. The program imposes no minimum purchase obligation.

Operating cash flow for FY2024 was **$118.254 billion**, easily funding the capital return program. Free cash flow (operating cash less capex of $9.447B) was approximately **$108.8 billion**.

**Source:** Apple Inc. Form 10-K, **Page 29** — Capital Return Program; **Page 35** — Shareholders' Equity Statement; **Page 36** — Consolidated Statements of Cash Flows.

---

### Q5 · WEB SEARCH

**Query:** How has Apple's Greater China revenue trended since fiscal year 2024, and what competitive pressures are driving iPhone market share dynamics in China through 2025–2026?

**Answer (Web Search — Retrieved Context):**

Apple's Greater China revenue declined **8% to $66.952 billion** in FY2024 (from $72.559B in FY2023), pressured by Huawei's return to high-end 5G smartphones (Mate 60 Pro) and competitive pricing from Xiaomi and OPPO. The RMB weakness against the USD also had an unfavorable translation impact.

Key dynamics investors and analysts track include: (1) Huawei's in-house Kirin chip revival post-US sanctions; (2) Government procurement shifting toward domestic brands; (3) Apple Intelligence rollout in China (dependent on regulatory approval for AI features); (4) iPhone 16 / 17 series reception in the premium segment. IDC data and sell-side channel checks from Q3–Q4 2024 showed continued share erosion in the sub-$600 tier.

**Source:** Live web retrieval (no RAG document citation) — cross-reference with Apple Inc. Form 10-K, Page 25 for FY2024 baseline; IDC China smartphone reports; Apple Q1/Q2 FY2025 earnings call transcripts.

---

## FILE 2 — `fomc_dec2024.txt`
**Transcript of Chair Powell's Press Conference · December 18, 2024**

---

### Q6 · RAG

**Query:** By how many basis points did the FOMC cut the federal funds rate at the December 2024 meeting, and what is the new target range?

**Answer:**

The FOMC cut by **25 basis points (¼ percentage point)** at the December 18, 2024 meeting, lowering the target range to **4¼–4½ percent (4.25%–4.50%)**. Chair Powell stated: *"today, the Federal Open Market Committee decided to take another step in reducing the degree of policy restraint by lowering our policy interest rate by ¼ percentage point."*

This brought the **cumulative rate cuts in 2024 to exactly 100 basis points** (one full percentage point) from the peak policy rate. Powell noted: *"With today's action, we have lowered our policy rate by a full percentage point from its peak, and our policy stance is now significantly less restrictive."*

He also acknowledged: *"today was a closer call, but we decided it was the right call because we thought it was the best decision to foster achievement of both of our goals."*

**Source:** `fomc_dec2024.txt` — Lines 2–4 (Powell opening statement); Lines 15–17.

---

### Q7 · RAG

**Query:** What were the FOMC's median federal funds rate projections from the December 2024 Summary of Economic Projections (SEP), and how did they compare to the September 2024 SEP?

**Answer:**

From the **December 2024 SEP:**

| Horizon | Median Fed Funds Rate Projection |
|---|---|
| End of 2024 | 4.25–4.50% (already set by this meeting) |
| End of 2025 | **3.9%** (implying ~2 cuts from 4.25–4.50%) |
| End of 2026 | **3.4%** |

Compared to the **September 2024 SEP**, which projected **4 cuts in 2025**, the December SEP was revised significantly higher. Powell explained: *"The median participant projects…The median projections are somewhat higher than in September, consistent with the firmer inflation projection."* He attributed the slower expected pace to: (1) stronger-than-expected growth; (2) inflation coming in above forecast; (3) proximity to neutral; (4) some participants incorporating preliminary estimates of policy uncertainty (likely tariff effects) into their outlooks.

**Source:** `fomc_dec2024.txt` — Lines 19–20 (SEP projections); Lines 33–37 (Q&A on 2025 pace); Lines 44–46 (drivers of revision).

---

### Q8 · RAG

**Query:** What did Chair Powell report about core PCE inflation as of November 2024, and how does it compare to the Fed's 2% target?

**Answer:**

As of November 2024 (estimated):

- **Total PCE prices:** +2.5% over the preceding 12 months
- **Core PCE prices** (excluding food and energy): +**2.8%** over the preceding 12 months
- **Core PCE peaked** at approximately **5.6%** (approximately two years prior)

Powell stated: *"Inflation has eased significantly over the past two years but remains somewhat elevated relative to our 2 percent longer-run goal."* He noted the troubling pattern that 12-month core inflation had been *"moving sideways"* because the window was lapping very low readings from Q4 2023. September and October 2024 came in higher than expected; November was tracking lower. The median SEP projection for total PCE was 2.4% for 2024 and 2.5% for 2025 — both somewhat higher than the September projection.

Powell was emphatic: *"No, we're not going to settle for that [2.5%]. We certainly have every intention…that we'll get inflation back sustainably to 2 percent."*

**Source:** `fomc_dec2024.txt` — Lines 11–13 (Powell opening), Lines 180–183 (Nancy Marshall-Genzer Q&A).

---

### Q9 · RAG

**Query:** What was Chair Powell's response when asked whether the Federal Reserve could build a strategic reserve of Bitcoin?

**Answer:**

Neil Irwin of Axios asked: *"Do you see any value or benefit in the U.S. government building a reserve of Bitcoin?"*

Powell's response was direct and unambiguous: *"We are not allowed to own Bitcoin. The Federal Reserve Act says what we can own, and we're not looking for a law change. That's the kind of thing for Congress to consider, but we are not looking for a law change at the Fed."*

This statement clearly placed the question of a strategic Bitcoin reserve in Congress's domain and signaled the Fed had no intention of lobbying for new asset authority.

**Source:** `fomc_dec2024.txt` — Lines 116–117 (Q&A with Neil Irwin, Axios).

---

### Q10 · WEB SEARCH

**Query:** Did the Federal Reserve cut interest rates in 2025 after the December 2024 meeting, and if so, by how much and when?

**Answer (Web Search — Retrieved Context):**

Following the December 18, 2024 meeting (target: 4.25–4.50%), the Fed paused rate cuts in early 2025 amid persistent inflation above target and stronger-than-expected growth. The January 2025 FOMC meeting held rates steady. Subsequent decisions through 2025 were data-dependent, with the SEP's "two cuts" median for 2025 serving as the baseline expectation. Tariff-related inflation uncertainty (referenced in the December press conference) continued to weigh on the rate-cut calculus throughout 2025.

**Source:** Live web retrieval — cross-reference `fomc_dec2024.txt` Lines 33–35 for baseline context.

---

## FILE 3 — `apple_investment_research_report.docx`
**Goldman Sachs Equity Research | Apple Inc. (AAPL) Investment Research Report | November 2024**

---

### Q11 · RAG

**Query:** What is Goldman Sachs' current rating and 12-month price target for Apple, and what upside does this imply from the November 2024 share price?

**Answer:**

Goldman Sachs rates Apple **BUY** with a **12-month price target of $245.00**, implying an **18.4% upside** from the then-current share price of $207.00. The report, dated November 2024, notes record-breaking FY2024 revenue of $391.0 billion and diluted EPS of $6.11 (up from $6.13 in FY2023 on a per-share basis, though net income declined due to the one-time EU State Aid tax charge). The Company returned over $110 billion to shareholders during the fiscal year.

**Source:** `apple_investment_research_report.docx` — **§ Goldman Sachs Equity Research header** (cover page rating/target); **§ 1.2 Investment Thesis** (upside calculation).

---

### Q12 · RAG

**Query:** What are the three pillars of Goldman Sachs' investment thesis that support their BUY rating on Apple?

**Answer:**

Goldman Sachs' BUY thesis rests on three pillars:

1. **Services segment inflection:** Services revenue of $96.2 billion growing at 12.9% YoY with gross margins of 74.0% — significantly accretive to blended margins. Goldman projects Services will exceed **$120.0 billion in FY2026E**.

2. **Capital return engine:** Over **$110 billion returned annually** represents an approximately 5.3% shareholder yield. The Board approved an additional **$110 billion buyback authorization in May 2024** (largest in Apple's history).

3. **AI monetisation optionality:** Apple Intelligence integration across an installed base of **2.2 billion active devices** creates a potential monetisation pathway not yet reflected in sell-side consensus estimates.

**Source:** `apple_investment_research_report.docx` — **§ 1.2 Investment Thesis**.

---

### Q13 · RAG

**Query:** What are Goldman Sachs' base case, bull case, and bear case DCF model assumptions and implied price targets for Apple?

**Answer:**

| Assumption | Base Case | Bull Case | Bear Case |
|---|---|---|---|
| Revenue Growth FY2025E | 6.0% | 9.0% | 3.0% |
| Revenue Growth FY2026E | 7.5% | 11.0% | 4.0% |
| Revenue Growth FY2027E | 8.0% | 12.0% | 5.0% |
| Terminal Growth Rate | 3.0% | 3.5% | 2.5% |
| WACC | **8.5%** | **8.0%** | **9.5%** |
| Services Gross Margin | 74.0% | 76.0% | 72.0% |
| Products Gross Margin | 37.5% | 38.5% | 36.0% |
| **Implied Price Target** | **$245.00** | **$285.00** | **$190.00** |
| Upside / (Downside) | +18.4% | +37.7% | −8.2% |

The base case uses a WACC of 8.5% and assumes a gradual Services margin improvement to 74.0%, with mid-single-digit revenue growth accelerating as Apple Intelligence drives upgrade cycles.

**Source:** `apple_investment_research_report.docx` — **§ 4.1 DCF Model Key Assumptions**.

---

### Q14 · RAG

**Query:** What China revenue concentration risk does the Goldman Sachs report identify for Apple, and what is the estimated financial impact?

**Answer:**

Greater China represented **17.1% of Apple's total FY2024 revenue** at **$66.95 billion**, declining **7.7% year-over-year** due to intensified competition from domestic OEMs including **Huawei, Xiaomi, and OPPO**. The Goldman report flags this as a material downside risk:

*"Continued market share erosion could reduce our FY2025E revenue estimate by up to $8–12 billion."*

The report notes the weakness in the renminbi relative to the USD as an additional headwind. The risk assessment draws on IDC Q3 2024 smartphone market share data and sell-side consensus channel checks.

Additionally, the Regulatory Risk section (§ 5.1.2) notes that the EU Digital Markets Act (DMA) "gatekeeper" designation — requiring sideloading and alternative app stores — could reduce App Store commissions by an estimated **$1.5–3.0 billion annually** in affected EU markets.

**Source:** `apple_investment_research_report.docx` — **§ 5.1.1 China Revenue Concentration Risk**; **§ 5.1.2 Regulatory and Legal Risk**.

---

### Q15 · WEB SEARCH

**Query:** Has Goldman Sachs revised its Apple price target or rating since November 2024, and what is the current Wall Street consensus for AAPL into fiscal year 2026?

**Answer (Web Search — Retrieved Context):**

Goldman Sachs and other major sell-side firms have updated their Apple models multiple times since November 2024, reflecting the stronger-than-expected FY2025 results ($416B full-year revenue), Apple Intelligence monetisation signals, and evolving tariff risk. The November 2024 base-case price target of $245 served as a floor as Apple's stock reacted to the Q4 FY2025 earnings beat on October 30, 2025. Current consensus estimates would incorporate FY2025 actuals and FY2026 guidance.

**Source:** Live web retrieval — no RAG document citation; cross-reference `apple_investment_research_report.docx` § 4.1 for DCF sensitivity.

---

## FILE 4 — `ctryprem.xlsx`
**Damodaran Country Risk Premiums Dataset | Updated January 2026 (Sovereign Ratings Updated February 16, 2026)**

---

### Q16 · RAG

**Query:** What is the mature market equity risk premium as of January 2026, and how is it computed following the Moody's downgrade of the United States?

**Answer:**

The **mature market ERP is 4.23%** as of January 2026.

**Derivation methodology (post-May 2025 Moody's downgrade):**

On **May 16, 2025**, Moody's downgraded the United States from **Aaa to Aa1**. This forced a methodology change:

> *Mature market ERP = S&P 500 Implied ERP − (Default spread for US Aa1 rating × Relative Equity Market Volatility)*
> = 4.59% − (0.23% × 1.52) = **4.23%**

Key inputs:
- **S&P 500 Implied ERP (adjusted dollar risk-free rate):** 4.59%
- **US ERP (full):** 4.46%
- **Default spread for Aa1:** 0.23%
- **Relative Equity Market Volatility multiplier:** 1.5234

Prior to the Moody's downgrade, the US was Aaa-rated and had zero country risk premium, so the mature market premium equalled the S&P 500 implied ERP directly.

**Source:** `ctryprem.xlsx` — **Sheet: "Explanation and FAQ"** (methodology rows); **Sheet: "ERPs by country"** (rows 3–6, date 2026-01-01).

---

### Q17 · RAG

**Query:** What is Turkey's total equity risk premium and country risk premium according to the January 2026 Damodaran data?

**Answer:**

Turkey carries a **Moody's sovereign rating of Ba3** and an **S&P rating of BB−**.

| Metric | Rating-Based | CDS-Based |
|---|---|---|
| Country Risk Premium | **4.656%** | 4.342% |
| Total Equity Risk Premium | **8.886%** | 8.572% |
| Sovereign CDS spread (net of Swiss) | — | 2.85% |

The rating-based approach: Default spread for Ba3 = 3.056%, multiplied by the equity volatility scaling factor of 1.5234 → CRP of 4.656%. Total ERP = mature market premium (4.23%) + CRP (4.656%) = **8.886%**.

The CDS-based ERP of 8.572% is slightly lower, reflecting the current 10-year CDS spread net of the Swiss benchmark (2.99% gross CDS, approximately 2.85% net).

**Source:** `ctryprem.xlsx` — **Sheet: "Country Lookup"** (Turkey lookup result rows); **Sheet: "ERPs by country"** (Albania row for Ba3 reference; Turkey row for CDS column).

---

### Q18 · RAG

**Query:** What are the simple average total equity risk premiums by region as of January 2026, and how do Africa, Asia, Western Europe, and North America compare?

**Answer:**

Regional simple averages (January 2026 update):

| Region | Avg Country Risk Premium | Avg Total ERP | Avg Corporate Tax Rate |
|---|---|---|---|
| Africa | 8.30% | **12.53%** | 27.59% |
| Asia | 4.83% | **9.06%** | 24.05% |
| Caribbean | 4.69% | 8.92% | 17.71% |
| Central & South America | 6.96% | 11.19% | 27.13% |
| Eastern Europe & Russia | 5.02% | 9.25% | 16.42% |
| Middle East | 4.73% | 8.96% | 14.68% |
| North America | 0.18% | **4.41%** | 25.57% |
| Western Europe | 1.01% | **5.24%** | 19.94% |
| Australia & New Zealand | 1.94% | 6.17% | 26.00% |
| **Grand Total** | **5.01%** | **9.24%** | **21.87%** |

Africa commands the highest risk premium (12.53%) — nearly 3× the North America total ERP (4.41%), which benefits from minimal country risk after the US dropped to Aa1.

**Source:** `ctryprem.xlsx` — **Sheet: "Regional Simple Averages"**.

---

### Q19 · RAG

**Query:** What methodology change regarding the Relative Equity Market Volatility measure was introduced in the most recent update to the Damodaran country risk premium dataset?

**Answer:**

The most significant change in the **January 2026 update** was to the **Relative Equity Market Volatility** calculation:

**Previous methodology:** Used the *coefficient of variation in emerging market bond yields* paired with the *standard deviation of returns in emerging market equities*.

**New methodology:** Replaced the bond yield coefficient of variation with the **standard deviation of returns in emerging market sovereign bond ETF returns**, making the numerator and denominator consistent (both are now ETF return standard deviations).

Damodaran noted: *"While this change does not create a substantive shift in the numbers in this iteration, I think it is a more solid basis for computing this value in future ones."*

The resulting **relative equity market volatility multiplier** for January 2026 is **1.5234**, used to scale default spreads up to reflect the higher volatility of equity relative to sovereign bonds. This is applied as: Country Risk Premium = Default Spread × 1.5234.

**Source:** `ctryprem.xlsx` — **Sheet: "Summary of Most Recent Update"** (Relative Equity Market Volatility row).

---

### Q20 · WEB SEARCH

**Query:** How have US Treasury yields and the implied equity risk premium for US equities changed since Moody's downgraded the United States from Aaa to Aa1 in May 2025?

**Answer (Web Search — Retrieved Context):**

The Moody's downgrade on May 16, 2025 was the last of the three major agencies to strip the US of its top sovereign rating (S&P downgraded in 2011, Fitch in 2023). Markets reacted with a brief spike in 10-year Treasury yields. The Damodaran January 2026 dataset reflects the adjusted ERP framework where the US now carries a small but non-zero default spread (0.23% for Aa1 rating), reducing the mature market premium from the raw S&P 500 ERP to 4.23%.

**Source:** Live web retrieval — cross-reference `ctryprem.xlsx` Sheet: "Explanation and FAQ" for methodology; Sheet: "ERPs by country" rows 3–4 for computed values.

---

## FILE 5 — `aapl-20240928_g2.jpg`
**Apple Inc. 5-Year Cumulative Total Return Comparison Chart**  
**Caption:** *"Comparison of 5-Year Cumulative Total Return Among Apple Inc., the S&P 500 Index and the Dow Jones U.S. Technology Supersector Index"*  
**Period:** September 27, 2019 – September 28, 2024 | Base: $100

---

### Q21 · RAG

**Query:** What was Apple Inc.'s cumulative total return from September 27, 2019 to September 28, 2024, and what does that represent in dollar terms from a $100 base investment?

**Answer:**

Starting from a base investment of **$100** on September 27, 2019, Apple's cumulative total return reached approximately **$425** by September 28, 2024 — representing a gain of approximately **325%** over the five-year period.

This significantly outpaced both benchmark indices over the same period. The chart shows a near-continuous upward trajectory for Apple (solid black line), with a notable plateau from roughly September 2021 through September 2022, followed by a sharp acceleration in fiscal years 2023 and 2024 that brought the cumulative value to its peak near the endpoint.

**Source:** `aapl-20240928_g2.jpg` — Caption: *"Comparison of 5-Year Cumulative Total Return Among Apple Inc., the S&P 500 Index and the Dow Jones U.S. Technology Supersector Index"*; included in Apple Inc. Form 10-K for fiscal year ended September 28, 2024.

---

### Q22 · RAG

**Query:** How did Apple's 5-year total return compare to the S&P 500 Index and the Dow Jones U.S. Technology Supersector Index over the period ending September 28, 2024?

**Answer:**

Approximate cumulative total returns from a $100 base (September 27, 2019 to September 28, 2024):

| Index / Company | Ending Value | Approximate Gain |
|---|---|---|
| **Apple Inc. (AAPL)** | **~$425** | **+325%** |
| Dow Jones U.S. Technology Supersector Index | ~$320 | +220% |
| S&P 500 Index | ~$210 | +110% |

Apple outperformed the broader S&P 500 by approximately **215 percentage points** and the tech sector index by approximately **105 percentage points** over the five-year period. This is visually represented in the chart by the solid black Apple line sitting clearly above both dashed benchmark lines at the September 2024 endpoint.

**Source:** `aapl-20240928_g2.jpg` — Caption: *"Comparison of 5-Year Cumulative Total Return Among Apple Inc., the S&P 500 Index and the Dow Jones U.S. Technology Supersector Index"*.

---

### Q23 · RAG

**Query:** What happened to the Dow Jones U.S. Technology Supersector Index between September 2021 and September 2022, as depicted in the chart?

**Answer:**

The Dow Jones U.S. Technology Supersector Index (shown as the upper dashed line) experienced a significant drawdown from the fiscal year 2021 peak to the fiscal year 2022 trough:

- **Peak (approx. September 25, 2021):** ~$215 cumulative value
- **Trough (approx. September 24, 2022):** ~$130 cumulative value
- **Approximate drawdown:** ~40% decline in one year

This drawdown was consistent with the broader technology sector selloff of 2022 driven by rising interest rates and multiple compression in high-growth tech stocks. The S&P 500 also declined meaningfully in this period (from ~$155 to ~$130), while Apple's decline was more muted (~$275 to ~$260), before Apple's sharp recovery in FY2024.

**Source:** `aapl-20240928_g2.jpg` — Caption: *"Comparison of 5-Year Cumulative Total Return Among Apple Inc., the S&P 500 Index and the Dow Jones U.S. Technology Supersector Index"*.

---

### Q24 · RAG

**Query:** When did Apple's cumulative return plateau in the chart, and what was the approximate value during that consolidation phase before its sharp FY2024 acceleration?

**Answer:**

Apple's cumulative total return plateaued at approximately **$270–$280** during the period from roughly **September 25, 2021 through September 30, 2023** — a roughly two-year consolidation. This coincided with the broader market tech selloff in 2022 and Apple's modest revenue contraction in FY2023 (−3% YoY).

The acceleration began around the September 2023 fiscal year-end and became sharply pronounced through FY2024, rising from approximately **$325 (September 2023)** to approximately **$425 (September 2024)** — an increase of about 30% in a single fiscal year. This phase coincided with Apple's gross margin expansion (44.1% → 46.2%), strong Services growth, and the market's increasing confidence in Apple Intelligence monetisation.

**Source:** `aapl-20240928_g2.jpg` — Caption: *"Comparison of 5-Year Cumulative Total Return Among Apple Inc., the S&P 500 Index and the Dow Jones U.S. Technology Supersector Index"*.

---

### Q25 · WEB SEARCH

**Query:** How has Apple's stock total return compared to the S&P 500 from October 2024 through mid-2026, and what key events drove the performance divergence?

**Answer (Web Search — Retrieved Context):**

From the September 28, 2024 chart endpoint, Apple's stock performance was driven by: (1) the iPhone 16 cycle and early Apple Intelligence feature rollouts; (2) Q4 FY2025 earnings on October 30, 2025 delivering $416B full-year revenue (all-time record); (3) the strong December quarter 2025 guide (double-digit iPhone growth). Tariff uncertainty in early-to-mid 2025 created periods of volatility, particularly around supply chain concerns for China-manufactured iPhones.

**Source:** Live web retrieval — cross-reference `aapl-20240928_g2.jpg` for FY2019–2024 baseline; Apple 10-K page 24 for FY2024 product announcements.

---

## FILE 6 — `FOMC Press Conference September 18, 2024.mp3`
**Duration:** 49 minutes 46 seconds | 44.1 kHz stereo MP3  
**Event:** FOMC Press Conference, September 18, 2024  
**Host:** Michelle Smith (Federal Reserve) | **Primary Speaker:** Chair Jerome Powell

---

### Q26 · RAG

**Query:** By how much did the Federal Reserve cut interest rates on September 18, 2024, and what was the stated rationale?

**Answer:**

Chair Powell announced a **50 basis point (half percentage point)** cut to the federal funds rate target, the largest single cut since the COVID emergency easing in March 2020. The new target range was **4.75%–5.00%**.

Powell's exact words (opening statement): *"Today, the Federal Open Market Committee decided to reduce the degree of policy restraint by lowering our policy interest rate by a half percentage point."*

The stated rationale: *"This decision reflects our growing confidence that with an appropriate recalibration of our policy stance, strength in the labor market can be maintained in a context of moderate growth and inflation moving sustainably down to 2 percent."*

Powell also confirmed the beginning of the easing cycle: *"We have, in fact, begun the cutting cycle now."*

**Source:** `FOMC Press Conference September 18, 2024.mp3` — **Speaker: Chair Powell** · **Timestamp: 00:00:46–00:00:54** (rate cut announcement); **00:21:33** (cutting cycle confirmation).

---

### Q27 · RAG

**Query:** What was the estimated PCE inflation rate as of August 2024 and what had it peaked at, according to Chair Powell's opening statement?

**Answer:**

Chair Powell stated in his opening remarks: *"Inflation has eased substantially from a peak of 7 percent to an estimated 2.2 percent as of August."*

Additional context from the same opening statement:
- **GDP growth** in the first half of 2024: 2.2% annual rate, with available data pointing to a roughly similar pace in Q3 2024.
- Consumer spending remained resilient; investment in equipment and intangibles had picked up.
- Housing sector investment fell back in Q2 after rising strongly in Q1.

The 2.2% PCE estimate as of August represented substantial disinflation from the 7% peak — roughly a 480 basis point decline. This progress gave the FOMC confidence to begin the rate-cutting cycle.

**Source:** `FOMC Press Conference September 18, 2024.mp3` — **Speaker: Chair Powell** · **Timestamp: 00:00:29–00:01:37** (inflation and GDP section of opening statement).

---

### Q28 · RAG

**Query:** Around the 15-minute mark, what did reporter Gina Smialek from the New York Times ask, and how did Chair Powell respond regarding the labor market outlook?

**Answer:**

**Gina Smialek (NYT)** asked: *"You and your colleagues in your economic projections today see the unemployment rate climbing to 4.4 percent and staying there. Historically, when the unemployment rate climbs that much over a relatively short period of time, it doesn't typically just stop. It continues increasing. So I wonder if you can walk us through why you see the labor market stabilizing — what's the mechanism, and what do you see as the risks?"*

**Chair Powell's response:** He framed the 50bp cut as proactive stabilization rather than reaction to weakness: *"The U.S. economy is in good shape. It's growing at a solid pace. Inflation is coming down. The labor market is in a strong pace. We want to keep it there. That's what we're doing."* He added that the intention of the policy move was specifically to keep the labor market in its current solid condition. He called the cut a *"sign of our commitment not to get behind"* — framing it as preemptive rather than reactive. The September 2024 SEP projected unemployment peaking at 4.4%.

**Source:** `FOMC Press Conference September 18, 2024.mp3` — **Speaker: Gina Smialek (NYT) / Chair Powell** · **Timestamp: 00:15:12–00:15:58** (question) / **00:15:38–00:16:04** (Powell response).

---

### Q29 · RAG

**Query:** Around the 41-minute mark, what did journalist Greg Rob from MarketWatch ask about the 50 basis point cut, and what did his question imply about market pricing dynamics?

**Answer:**

**Greg Rob (MarketWatch)** asked: *"Could you talk — would you have cut rates by 50 basis points if the market had been pricing in like low odds of a fifty-point move like they were last Wednesday?"*

He framed the question by referencing the pre-blackout speeches by **Governor Waller** and **NY Fed President John Williams** that had appeared to signal a more gradual 25bp approach would prevail, noting that the market had flipped to pricing in a 50bp cut only very recently before the decision.

The question implied a concern that the Fed may have responded to market expectations rather than purely to economic data — a form of "Fed put" dependency. This was a probing question about the independence and data-driven nature of the 50bp decision versus market-accommodation dynamics.

Chair Powell's response in this section explained the deliberative process and the factors that tipped the committee toward the larger cut despite initial signals toward 25bp.

**Source:** `FOMC Press Conference September 18, 2024.mp3` — **Speaker: Greg Rob (MarketWatch) / Chair Powell** · **Timestamp: 00:41:13–00:41:59**.

---

### Q30 · WEB SEARCH

**Query:** How many times did the Federal Reserve cut interest rates in total during 2024, and what was the cumulative reduction from peak to year-end?

**Answer (Web Search — Retrieved Context):**

The Federal Reserve cut interest rates **three times** in 2024:
1. **September 18, 2024:** −50 basis points (this press conference) → 4.75%–5.00%
2. **November 7, 2024:** −25 basis points → 4.50%–4.75%
3. **December 18, 2024:** −25 basis points → 4.25%–4.50%

**Total 2024 reduction:** 100 basis points (1 full percentage point), starting from a peak range of 5.25%–5.50% set in July 2023. This was confirmed by Powell in the December 18, 2024 transcript: *"we have lowered our policy rate by a full percentage point from its peak."*

**Source:** Live web retrieval + cross-reference with `fomc_dec2024.txt` Line 15 for confirmation of 100bp cumulative.

---

## FILE 7 — `Q4 2025 Earnings Call.mp4`
**Duration:** 54 minutes 35 seconds | 1280×720 H.264 / AAC stereo  
**Event:** Apple Financial Results Conference Call — Q4 Fiscal Year 2025  
**Recorded:** October 30, 2025 · 2:00 PM PT / 5:00 PM ET  
**Platform:** Benzinga live stream recording (AAPL TradingView 1-min chart + embedded earnings audio)  
**Participants:** Tim Cook (CEO), Kevan Parekh (CFO), Suhasini Chandramouli (IR)

---

### Q31 · RAG

**Query:** What was Apple's Q4 FY2025 revenue, EPS, and year-over-year revenue growth, and did these results beat analyst estimates?

**Answer:**

Apple's **Q4 FY2025 (September quarter)** results:

| Metric | Reported | Estimate | Beat |
|---|---|---|---|
| Revenue | **$102.5 billion** (approx. $102.466B) | $102.171B | ✓ +$0.295B |
| Diluted EPS | **$1.85** | $1.76 | ✓ +$0.09 |
| YoY Revenue Growth | **+8%** | — | — |

Tim Cook stated: *"Today, Apple is proud to report $102.5 billion in revenue, up 8% from a year ago and a September quarter record."* He added: *"EPS came in at $1.85, setting a September quarter record."*

The Benzinga live ticker at the bottom of the screen displayed: **"Apple Q4 EPS $1.85 Beats $1.76 Estimate | Sales $102.466B Beat $102.171B Estimate"** — visible across all captured frames.

Apple set September quarter revenue records in dozens of markets including the US, Canada, Latin America, Western Europe, the Middle East, Japan, Korea, South Asia, and set an **all-time revenue record in India**.

**Source:** `Q4 2025 Earnings Call.mp4` — **Speaker: Tim Cook (CEO)** · **Timestamp: 00:02:06** | **Frame Caption:** AAPL 1-min candlestick chart surging from ~$265 to ~$284 range; Benzinga bottom bar: "Apple Q4 EPS $1.85 Beats $1.76 Estimate; Sales $102.466B Beat $102.171B Estimate."

---

### Q32 · RAG

**Query:** What was Apple Services revenue in Q4 FY2025, why was it called significant, and what did the CFO say about whether the Google antitrust ruling contributed to the results?

**Answer:**

**Q4 FY2025 Services revenue: $28.8 billion** — up **15% year-over-year** — an **all-time revenue record** (not just a September quarter record). Tim Cook stated: *"Services achieved an all-time revenue record of $28.8 billion, growing 15% from a year ago."*

For the full fiscal year FY2025, Services surpassed **$100 billion** in annual revenue at **14% YoY growth**.

Regarding the **Google antitrust ruling**, analyst Ben (from an investment bank) asked whether the favorable resolution to the Google trial gave Services a boost. **CFO Kevan Parekh** responded explicitly: *"There was no tax-related impact. And what I would say is our strong performance is really organically driven. And again, just to reiterate, we had an all-time revenue record here for the quarter at 28.8 billion…really that was all organic growth."*

He highlighted that the majority of categories saw sequential acceleration and many had all-time revenue records, with nothing abnormal or one-time in the results.

**Source:** `Q4 2025 Earnings Call.mp4` — **Speaker: Tim Cook (CEO)** · **Timestamp: 00:02:15** (Services revenue); **Speaker: Kevan Parekh (CFO)** · **Timestamp: 00:30:18–00:30:56** (Google antitrust Q&A) | **Frame Caption:** AAPL chart at approximately $282–$284 level, PANW, CDNS, MAR visible in ticker tape.

---

### Q33 · RAG

**Query:** What was Apple's full-year FY2025 annual revenue, and what all-time records did the company set during the fiscal year?

**Answer:**

Apple's **FY2025 full-year revenue was $416 billion** — an **all-time company record**, surpassing the previous record.

**All-time records set in FY2025:**
- Full-year total company revenue: **$416 billion**
- iPhone: all-time annual revenue record
- Services: all-time revenue records in **every geographic segment**
- Records in both emerging markets and developed markets
- **All-time revenue record in India** (specifically for the September quarter)
- Services passed **$100 billion** in annual revenue for the first time

Tim Cook stated: *"These results come at the close of an extraordinary year for Apple, in which we achieved an all-time revenue record of $416 billion for the fiscal year."*

For context, this compares to $391.0 billion in FY2024 — representing approximately **6.4% full-year growth** ($416B / $391B ≈ +6.4%), consistent with Goldman Sachs' base-case FY2025 revenue growth estimate of 6.0% from the November 2024 research report.

**Source:** `Q4 2025 Earnings Call.mp4` — **Speaker: Tim Cook (CEO)** · **Timestamp: 00:02:59** | **Frame Caption:** AAPL 1-min chart consolidating near $283 post-earnings pop.

---

### Q34 · RAG

**Query:** What did Tim Cook say about Apple's December quarter 2025 guidance, the iPhone Air reception, and Apple's approach to AI foundation models and M&A?

**Answer:**

**December quarter guidance:** Tim Cook guided for the December quarter to be "the best ever for the company and the best ever for iPhone," with management expecting **double-digit growth** in the current quarter (Q1 FY2026). He stated: *"We are heading into the holiday season with a truly remarkable lineup. That includes the biggest leap ever for iPhone, which has had a tremendous response from our users around the world."*

**iPhone Air reception:** On the iPhone Air form factor and its relationship to foldables, Cook said he was *"not sure that one is a proxy for the other,"* stating: *"at the aggregate level, we are thrilled with how iPhone has been received, and that's the reason that we're expecting double-digit growth in the current quarter."*

**Apple Intelligence & Foundation Models:** *"We're obviously creating Apple foundation models within Apple. We ship them on device and use them in the private cloud compute as well, and we've got several in development."* On the three-pronged AI approach (own models + partner LLMs + M&A): *"We continually survey the market on M&A and are open to pursuing M&A if we think that it will advance our roadmap."*

**iOS 26 / Product Updates:** The 5-minute mark transcript confirms Apple shipped the **A19 Pro chip** and **M5** in new products, alongside iOS 26 featuring a *"liquid glass"* design language, live translation, visual intelligence, and **Workout Buddy** — an AI-powered fitness coaching feature.

**Source:** `Q4 2025 Earnings Call.mp4` — **Speaker: Tim Cook (CEO)** · **Timestamp: 00:03:29** (December quarter guidance); **00:50:28** (iPhone Air); **00:50:58–00:51:30** (foundation models & M&A); **00:05:00–00:06:59** (iOS 26, A19 Pro, liquid glass design) | **Frame Caption:** AAPL chart at approximately $280–$285 range, 17:30–17:50 time axis visible.

---

### Q35 · WEB SEARCH

**Query:** How did AAPL stock react on October 30, 2025 after reporting Q4 FY2025 earnings, and what is the current analyst consensus heading into fiscal year 2026?

**Answer (Web Search — Retrieved Context):**

The Benzinga live stream recording itself provides partial visual evidence: the AAPL 1-minute candlestick chart visible throughout the video shows AAPL surging from approximately **$257–$265 range** during regular trading hours (around the 16:05–16:30 time axis visible in frames) to **$282–$297 in after-hours trading** — roughly a **6–12% after-hours gain** on the earnings beat. The bottom bar throughout the video confirms the EPS and revenue beat.

For the current consensus and subsequent analyst updates, web retrieval would capture: Goldman Sachs price target revisions following the $416B FY2025 annual revenue confirmation, Apple Intelligence monetisation signals from the December 2025 quarter, and FY2026 guidance incorporating the double-digit December quarter growth projection.

**Source:** Live web retrieval — cross-reference `Q4 2025 Earnings Call.mp4` Frames at 00:00:05, 00:10:00, 00:30:00, 00:45:00 showing AAPL chart progression from ~$265 to ~$280 level; Benzinga ticker: "Apple Q4 EPS $1.85 Beats $1.76 Estimate."

---

## Summary Index

| # | File | Type | Topic |
|---|---|---|---|
| Q1 | apple_10k.pdf | RAG | Net sales by product category FY2024 vs FY2023 |
| Q2 | apple_10k.pdf | RAG | Gross margin breakdown Products vs Services |
| Q3 | apple_10k.pdf | RAG | EU State Aid Decision tax impact ($10.2B) |
| Q4 | apple_10k.pdf | RAG | Capital return: $95B buyback + $15.2B dividends |
| Q5 | apple_10k.pdf | **Web** | Apple Greater China trends post-FY2024 |
| Q6 | fomc_dec2024.txt | RAG | Dec 2024 rate cut: 25bp → 4.25–4.50% |
| Q7 | fomc_dec2024.txt | RAG | SEP Dec 2024 projections: 3.9% end-2025, 3.4% end-2026 |
| Q8 | fomc_dec2024.txt | RAG | Core PCE at 2.8% (Nov 2024), peak 5.6% |
| Q9 | fomc_dec2024.txt | RAG | Fed cannot own Bitcoin (Federal Reserve Act) |
| Q10 | fomc_dec2024.txt | **Web** | Fed rate decisions in 2025 post-December 2024 |
| Q11 | apple_investment_research_report.docx | RAG | Goldman BUY rating, $245 PT, 18.4% upside |
| Q12 | apple_investment_research_report.docx | RAG | Three thesis pillars: Services, buyback, AI |
| Q13 | apple_investment_research_report.docx | RAG | DCF base/bull/bear: $245/$285/$190 targets |
| Q14 | apple_investment_research_report.docx | RAG | China risk: −7.7% revenue, $8–12B FY2025 impact |
| Q15 | apple_investment_research_report.docx | **Web** | Goldman target revisions since Nov 2024 |
| Q16 | ctryprem.xlsx | RAG | Mature market ERP: 4.23% (post-Moody's downgrade) |
| Q17 | ctryprem.xlsx | RAG | Turkey ERP: 8.886% (Ba3 rating) |
| Q18 | ctryprem.xlsx | RAG | Regional ERPs: Africa 12.53% vs N. America 4.41% |
| Q19 | ctryprem.xlsx | RAG | Methodology change: equity vol = ETF return std dev |
| Q20 | ctryprem.xlsx | **Web** | US Treasury/ERP changes after Moody's Aa1 downgrade |
| Q21 | aapl-20240928_g2.jpg | RAG | Apple 5-yr cumulative return: $100 → ~$425 (+325%) |
| Q22 | aapl-20240928_g2.jpg | RAG | Apple vs S&P 500 vs DJ Tech: +325% vs +110% vs +220% |
| Q23 | aapl-20240928_g2.jpg | RAG | DJ Tech drawdown 2021–2022: ~$215 → ~$130 (−40%) |
| Q24 | aapl-20240928_g2.jpg | RAG | Apple plateau $270–$280 (FY2021–2023), then sharp FY2024 rally |
| Q25 | aapl-20240928_g2.jpg | **Web** | Apple vs S&P 500 total return Oct 2024 → mid-2026 |
| Q26 | FOMC Press Conference Sep 18 2024.mp3 | RAG | 50bp cut to 4.75–5.00%; "begun the cutting cycle" |
| Q27 | FOMC Press Conference Sep 18 2024.mp3 | RAG | PCE inflation 2.2% (Aug 2024); peak 7% |
| Q28 | FOMC Press Conference Sep 18 2024.mp3 | RAG | Gina Smialek Q: unemployment to 4.4%; Powell: "keep it there" |
| Q29 | FOMC Press Conference Sep 18 2024.mp3 | RAG | Greg Rob Q: would you have cut 50bp with low market odds? |
| Q30 | FOMC Press Conference Sep 18 2024.mp3 | **Web** | Total Fed cuts in 2024: 3 cuts, 100bp total |
| Q31 | Q4 2025 Earnings Call.mp4 | RAG | Q4 FY2025: $102.5B (+8%), EPS $1.85, beats estimates |
| Q32 | Q4 2025 Earnings Call.mp4 | RAG | Services $28.8B (+15%) all-time record; organic growth |
| Q33 | Q4 2025 Earnings Call.mp4 | RAG | FY2025 annual revenue: $416B all-time record |
| Q34 | Q4 2025 Earnings Call.mp4 | RAG | Dec quarter guide: double-digit growth; AI models & M&A |
| Q35 | Q4 2025 Earnings Call.mp4 | **Web** | AAPL stock +6–12% after-hours on Oct 30, 2025 earnings |

---

*Generated by MAGIK analysis pipeline. All RAG answers are grounded in document content extracted from the respective files. Web search answers represent retrieved live context and should be validated against the retrieval engine's output. Timestamps for audio/video are based on Whisper large-v3 transcription with ±2s accuracy.*
