"""
Generate RAG Evaluation PDF — 35 Query/Expected-Answer Pairs.
Usage: python scripts/generate_eval_pdf.py
Output: docs/RAG_Evaluation_35_Queries.pdf
"""
from __future__ import annotations

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

OUTPUT = "docs/RAG_Evaluation_35_Queries.pdf"

# ── colour palette ──────────────────────────────────────────────────────────
C_NAVY   = colors.HexColor("#1B2A4A")
C_ACCENT = colors.HexColor("#2E7D32")   # green for RAG
C_WEB    = colors.HexColor("#1565C0")   # blue for web-search
C_WARN   = colors.HexColor("#B71C1C")   # red for data limitation
C_LABEL  = colors.HexColor("#4A4A4A")
C_BG_RAG = colors.HexColor("#F1F8F1")
C_BG_WEB = colors.HexColor("#E8F0FE")
C_BORDER = colors.HexColor("#CCCCCC")
C_HEAD   = colors.HexColor("#ECEFF1")

# ── data ────────────────────────────────────────────────────────────────────
ENTRIES: list[dict] = [
    # ── File 1: aapl_10k_2023.txt ──────────────────────────────────────────
    {
        "q_num": 1, "source_file": "aapl_10k_2023.txt", "q_type": "RAG",
        "modality": "TXT",
        "query": "What were Apple's total net sales for fiscal year 2023, and how did they compare to fiscal year 2022?",
        "expected_answer": (
            "Apple's total net sales for FY2023 were $383,285 million — a decrease of $11,043 million "
            "(−2.8%) from FY2022's $394,328 million. Products net sales were $298,085M (vs. $316,199M in "
            "FY2022); Services net sales were $85,200M (vs. $78,129M in FY2022). Services grew while "
            "all major hardware categories declined."
        ),
        "citation": "Source: aapl_10k_2023.txt  |  Locator: none (flat TXT — no page/section locator)",
        "citation_type": None,
    },
    {
        "q_num": 2, "source_file": "aapl_10k_2023.txt", "q_type": "RAG",
        "modality": "TXT",
        "query": "What was Apple's net income and earnings per share (diluted) for fiscal year 2023?",
        "expected_answer": (
            "Apple's net income for FY2023 was $96,995 million. Diluted earnings per share (EPS) was "
            "$6.13 (basic EPS: $6.16). The diluted share count used in the calculation was approximately "
            "15,812,547 thousand shares. Apple also declared dividends and dividend equivalents of $0.94 "
            "per share during FY2023."
        ),
        "citation": "Source: aapl_10k_2023.txt  |  Locator: none (flat TXT — no page/section locator)",
        "citation_type": None,
    },
    {
        "q_num": 3, "source_file": "aapl_10k_2023.txt", "q_type": "RAG",
        "modality": "TXT",
        "query": "Which Apple product category had the largest year-over-year revenue decline in FY2023, and by how much?",
        "expected_answer": (
            "Mac had the largest decline: net sales fell 27% (−$10.8 billion) in FY2023 vs. FY2022, "
            "primarily due to lower laptop sales. Other hardware declines: iPhone −2% (−$4.9B, from "
            "lower non-Pro models); iPad −3% (−$1.0B); Wearables, Home and Accessories −3% (−$1.4B). "
            "Services was the only category with growth (+$7.1B, +9.1% YoY)."
        ),
        "citation": "Source: aapl_10k_2023.txt  |  Locator: none (flat TXT — no page/section locator)",
        "citation_type": None,
    },
    {
        "q_num": 4, "source_file": "aapl_10k_2023.txt", "q_type": "RAG",
        "modality": "TXT",
        "query": "How much cash and cash equivalents did Apple report on its balance sheet at the end of fiscal 2023?",
        "expected_answer": (
            "Apple reported cash and cash equivalents of $29,965 million ($~30.0B) at September 30, 2023 "
            "(end of FY2023). Total cash, cash equivalents, and restricted cash at period-end was "
            "$30,737 million. During FY2023, Apple repurchased $76.6 billion of common stock and paid "
            "$15.0 billion in dividends, while maintaining its strong balance sheet."
        ),
        "citation": "Source: aapl_10k_2023.txt  |  Locator: none (flat TXT — no page/section locator)",
        "citation_type": None,
    },
    {
        "q_num": 5, "source_file": "Web Search", "q_type": "WEB",
        "modality": "WEB",
        "query": "What is Apple's latest/most recent stock price and trailing twelve-month revenue? Search the internet for current data.",
        "expected_answer": (
            "As of June 9, 2026, Apple's share price is approximately $301.54. Trailing twelve-month "
            "(TTM) revenue for the period ending March 31, 2026 was $451.44 billion — a 12.76% YoY "
            "increase, the strongest growth rate since FY2021's post-pandemic surge."
        ),
        "citation": "Source: Web Search  |  Locator: internet retrieval (no local source file)",
        "citation_type": "web",
    },

    # ── File 2: aapl_def14a_2023.docx ──────────────────────────────────────
    {
        "q_num": 6, "source_file": "aapl_def14a_2023.docx", "q_type": "RAG",
        "modality": "DOCX",
        "query": "What was Tim Cook's total compensation package for fiscal year 2022, as reported in the Apple 2023 proxy statement?",
        "expected_answer": (
            "According to the 2023 DEF 14A proxy statement, Tim Cook received total compensation of "
            "$99.4 million for fiscal year 2022. Breakdown: base salary $3.0 million, stock awards "
            "$82.3 million, non-equity incentive compensation $13.0 million, and other compensation "
            "$1.1 million. The Compensation Committee determined this appropriately aligns pay with "
            "performance and shareholder interests."
        ),
        "citation": "Source: aapl_def14a_2023.docx  |  section_title: \"Executive Compensation\"",
        "citation_type": "docx",
    },
    {
        "q_num": 7, "source_file": "aapl_def14a_2023.docx", "q_type": "RAG",
        "modality": "DOCX",
        "query": "What performance metrics and targets were used to determine Apple's executive annual bonus in 2023?",
        "expected_answer": (
            "Apple's executive compensation programme (per 2023 proxy) consists of three elements: base "
            "salary, annual cash incentive, and long-term equity awards. The annual cash incentive is "
            "tied to financial performance — the proxy notes Apple's FY2022 net income was $99.8B, "
            "reflecting 'strong performance across all product categories and geographic segments,' and "
            "the Committee determined it 'appropriately aligns pay with performance and shareholder "
            "interests.' Since 2021, Apple added an ESG Modifier (±10% adjustment at Committee "
            "discretion) based on accomplishments in accessibility, education, environment, inclusion & "
            "diversity, privacy, and supplier responsibility."
        ),
        "citation": "Source: aapl_def14a_2023.docx  |  section_title: \"Executive Compensation\"",
        "citation_type": "docx",
    },
    {
        "q_num": 8, "source_file": "aapl_def14a_2023.docx", "q_type": "RAG",
        "modality": "DOCX",
        "query": "How many directors are on Apple's Board of Directors, and who chairs the board?",
        "expected_answer": (
            "Apple's Board has 9 directors total. Eight of the nine are independent. The proxy nominates: "
            "Tim Cook, James Bell, Al Gore, Andrea Jung, Art Levinson, Monica Lozano, Ronald Sugar, and "
            "Susan Wagner. Tim Cook serves as CEO (the sole non-independent director). Art Levinson is "
            "the Lead Independent Director and receives an additional annual retainer of $50,000. There "
            "is no separate non-executive Chairman role."
        ),
        "citation": "Source: aapl_def14a_2023.docx  |  section_title: \"Corporate Governance\"",
        "citation_type": "docx",
    },
    {
        "q_num": 9, "source_file": "aapl_def14a_2023.docx", "q_type": "RAG",
        "modality": "DOCX",
        "query": "What percentage of the executive compensation is tied to ESG or sustainability performance goals?",
        "expected_answer": (
            "Apple's 2023 proxy does not tie a fixed percentage of compensation to ESG metrics. Instead, "
            "since 2021 Apple applies an ESG Modifier that can adjust the annual cash incentive payout "
            "upward or downward by up to 10% — a qualitative, discretionary adjustment by the "
            "Compensation Committee based on Apple values (accessibility, education, environment, "
            "inclusion & diversity, privacy, and supplier responsibility). Apple is also committed to "
            "being carbon neutral across its full supply chain and product life cycle by 2030."
        ),
        "citation": "Source: aapl_def14a_2023.docx  |  section_title: \"Environmental, Social and Governance (ESG)\"",
        "citation_type": "docx",
    },
    {
        "q_num": 10, "source_file": "Web Search", "q_type": "WEB",
        "modality": "WEB",
        "query": "What is Tim Cook's latest/most recent estimated net worth and recent Apple stock sales activity in 2026? Search the internet.",
        "expected_answer": (
            "As of 2026, Tim Cook's estimated net worth is approximately $2.5–2.9 billion (Forbes: ~$2.5B; "
            "other sources up to $2.9B). He holds ~3.28 million AAPL shares worth ~$966M. Most recent "
            "trade: sale of 64,949 shares on April 2, 2026 (~$17M). All 8 transactions over 5 years are "
            "sells (0 buys). Cook announced he will step down as Apple CEO in September 2026 and plans to "
            "donate most of his wealth to philanthropy."
        ),
        "citation": "Source: Web Search  |  Locator: internet retrieval (no local source file)",
        "citation_type": "web",
    },

    # ── File 3: apple_q1_fy2025_earnings_commentary.mp3 ────────────────────────────
    {
        "q_num": 11, "source_file": "apple_q1_fy2025_earnings_commentary.mp3", "q_type": "RAG",
        "modality": "MP3",
        "query": "What was reported about iPhone revenue performance in the Apple earnings call audio?",
        "expected_answer": (
            "The CNBC-style earnings commentary reports iPhone revenue of $69.1 billion — described as "
            "'a rare miss' versus the $71.03 billion expected by analysts, and down slightly (less than "
            "1%) year-over-year. The audio is analyst commentary, not a direct Tim Cook quote. Total "
            "revenue was $124.3 billion (slight beat vs. $124.12B expected) and EPS was $2.40 (beat "
            "vs. $2.35 expected)."
        ),
        "citation": "Source: apple_q1_fy2025_earnings_commentary.mp3  |  timestamp_start: 0.0s  |  timestamp_end: 30.0s",
        "citation_type": "mp3",
    },
    {
        "q_num": 12, "source_file": "apple_q1_fy2025_earnings_commentary.mp3", "q_type": "RAG",
        "modality": "MP3",
        "query": "What were the overall Apple revenue and EPS results reported in the earnings call audio?",
        "expected_answer": (
            "The audio reports: EPS $2.40 (beat vs. $2.35 expected), total revenue $124.3 billion "
            "(slight beat vs. $124.12 billion expected), iPhone $69.1 billion (miss vs. $71.03B), "
            "Services $26.34 billion (beat vs. $26.09B, +14% YoY), and Greater China $18.5 billion "
            "(down 11%). The stock finished 'flat-ish' following the report."
        ),
        "citation": "Source: apple_q1_fy2025_earnings_commentary.mp3  |  timestamp_start: 0.0s  |  timestamp_end: 30.0s",
        "citation_type": "mp3",
    },
    {
        "q_num": 13, "source_file": "apple_q1_fy2025_earnings_commentary.mp3", "q_type": "RAG",
        "modality": "MP3",
        "query": "What did the analyst commentary say about Apple Services revenue in the earnings call audio?",
        "expected_answer": (
            "Services revenue came in at $26.34 billion, beating the $26.09 billion expected by "
            "analysts — up approximately 14% year-over-year. This was highlighted as a beat alongside "
            "the iPhone miss. The commentary notes Services as a key driver of Apple's consistent, "
            "less hit-driven revenue profile."
        ),
        "citation": "Source: apple_q1_fy2025_earnings_commentary.mp3  |  timestamp_start: 0.0s  |  timestamp_end: 30.0s",
        "citation_type": "mp3",
    },
    {
        "q_num": 14, "source_file": "apple_q1_fy2025_earnings_commentary.mp3", "q_type": "RAG",
        "modality": "MP3",
        "query": "Which geographic market was highlighted as a weak spot in the Apple earnings call audio, and by how much did it decline?",
        "expected_answer": (
            "Greater China was highlighted as the key weak spot: revenue declined 11% to $18.5 billion. "
            "No strong-growth emerging markets are named. The overall commentary focuses on the "
            "iPhone miss ($69.1B vs $71.03B expected) and Services beat ($26.34B, +14% YoY)."
        ),
        "citation": "Source: apple_q1_fy2025_earnings_commentary.mp3  |  timestamp_start: 0.0s  |  timestamp_end: 30.0s",
        "citation_type": "mp3",
    },
    {
        "q_num": 15, "source_file": "Web Search", "q_type": "WEB",
        "modality": "WEB",
        "query": "What were Wall Street analyst consensus estimates for Apple's Q4 FY2023 earnings per share and revenue? Search the internet for historical analyst estimates.",
        "expected_answer": (
            "For Apple's Q4 FY2023 (quarter ending September 30, 2023): consensus EPS estimate was ~$1.39; "
            "consensus revenue estimate was ~$89.28 billion. Actual results: EPS $1.46 (beat by $0.07, "
            "+5%); Revenue $89.5B (beat by ~$220M). Services revenue of $22.3B significantly beat the "
            "~$21.35B estimate. Full FY2023 revenue was $383.29B, down ~3% YoY."
        ),
        "citation": "Source: Web Search  |  Locator: internet retrieval (no local source file)",
        "citation_type": "web",
    },

    # ── File 4: berkshire_letter_2022.pdf ───────────────────────────────────
    {
        "q_num": 16, "source_file": "berkshire_letter_2022.pdf", "q_type": "RAG",
        "modality": "PDF",
        "query": "What was Berkshire Hathaway's total operating earnings in 2022?",
        "expected_answer": (
            "Berkshire Hathaway had record operating earnings of $30.8 billion in 2022 — described by "
            "Warren Buffett as 'a good year.' Operating earnings is Berkshire's preferred profitability "
            "metric, which excludes investment gains/losses. The letter warns readers that even this "
            "figure can be manipulated by management and calls earnings manipulation 'one of the shames "
            "of capitalism.'"
        ),
        "citation": "Source: berkshire_letter_2022.pdf  |  page_number: 6",
        "citation_type": "pdf",
    },
    {
        "q_num": 17, "source_file": "berkshire_letter_2022.pdf", "q_type": "RAG",
        "modality": "PDF",
        "query": "How much did Berkshire Hathaway spend on share repurchases during 2022?",
        "expected_answer": (
            "The 2022 annual letter states Berkshire repurchased 1.2% of its outstanding shares during "
            "2022, with Buffett noting 'a very minor gain in per-share intrinsic value' through these "
            "repurchases. Similar buybacks at Apple and American Express (Berkshire's significant "
            "investees) further increased Berkshire's percentage ownership. The letter does not state "
            "the exact dollar amount in its narrative (detailed figures are in annual report pages "
            "K-33–K-66; the 10-K separately reports ~$7.9B in buybacks for 2022)."
        ),
        "citation": "Source: berkshire_letter_2022.pdf  |  page_number: 5",
        "citation_type": "pdf",
    },
    {
        "q_num": 18, "source_file": "berkshire_letter_2022.pdf", "q_type": "RAG",
        "modality": "PDF",
        "query": "What did Warren Buffett specifically say about Berkshire's investment in Apple in the 2022 annual letter?",
        "expected_answer": (
            "Buffett noted that 'a very minor gain in per-share intrinsic value took place in 2022 "
            "through Berkshire share repurchases as well as similar moves at Apple and American Express, "
            "both significant investees.' He emphasized that as Apple repurchases its own shares, "
            "Berkshire's percentage ownership in Apple increases at no additional cost. He also "
            "celebrated Berkshire as the largest owner of eight major U.S. companies at year-end 2022, "
            "with Apple being the most prominent investee. He framed share repurchases as value-accretive "
            "for continuing shareholders when done below intrinsic value."
        ),
        "citation": "Source: berkshire_letter_2022.pdf  |  page_number: 5",
        "citation_type": "pdf",
    },
    {
        "q_num": 19, "source_file": "berkshire_letter_2022.pdf", "q_type": "RAG",
        "modality": "PDF",
        "query": "What was the insurance underwriting profit or loss for Berkshire's insurance operations in 2022?",
        "expected_answer": (
            "The letter discusses insurance primarily through the lens of float: Berkshire's insurance "
            "float grew from $147 billion to $164 billion in 2022, aided by the Alleghany Corporation "
            "acquisition (captained by Joe Brandon). Buffett states: 'With disciplined underwriting, "
            "these funds have a decent chance of being cost-free over time.' The float has grown "
            "8,000-fold since Berkshire's first P&C insurer purchase in 1967. The letter does not "
            "provide a specific underwriting profit/loss dollar figure in its narrative section "
            "(detailed P&L is in appendix pages K-33–K-66)."
        ),
        "citation": "Source: berkshire_letter_2022.pdf  |  page_number: 5",
        "citation_type": "pdf",
    },
    {
        "q_num": 20, "source_file": "Web Search", "q_type": "WEB",
        "modality": "WEB",
        "query": "What is Berkshire Hathaway's latest/most recent book value per share and its current largest stock holdings in 2026? Search the internet.",
        "expected_answer": (
            "As of Q1 2026 (March 31, 2026): Berkshire Hathaway Class A book value per share ≈ $521.62; "
            "Class B (BRK.B) book value per share ≈ $337.15. The five largest equity holdings are: "
            "American Express, Apple, Bank of America, Coca-Cola, and Chevron — representing ~61% of "
            "total equity portfolio fair value. Berkshire also holds Kraft Heinz and Occidental "
            "Petroleum under equity-method accounting."
        ),
        "citation": "Source: Web Search  |  Locator: internet retrieval (no local source file)",
        "citation_type": "web",
    },

    # ── File 5: cnbc_earnings_highlight.mp4 ────────────────────────────────
    {
        "q_num": 21, "source_file": "cnbc_earnings_highlight.mp4", "q_type": "RAG",
        "modality": "MP4",
        "query": "What company's earnings results were featured in the CNBC earnings highlight video?",
        "expected_answer": (
            "The CNBC earnings highlight video features Apple Inc. (AAPL) Q4 FY2024 earnings results "
            "(for the quarter ending September 2024 — Apple's 'September quarter'). The segment includes "
            "on-air reporters, an analyst, and anchor John discussing the just-released results."
        ),
        "citation": "Source: cnbc_earnings_highlight.mp4  |  timestamp_start: 0.0s  |  timestamp_end: 15.0s",
        "citation_type": "mp4",
    },
    {
        "q_num": 22, "source_file": "cnbc_earnings_highlight.mp4", "q_type": "RAG",
        "modality": "MP4",
        "query": "What key earnings figure or headline number was reported in the CNBC video?",
        "expected_answer": (
            "Headline figures reported: (1) Adjusted EPS $1.64 — beat vs. $1.60 expected (there is a "
            "one-time EU tax charge, hence 'adjusted'); (2) Revenue $94.93B — beat vs. ~$94.5B expected; "
            "(3) iPhone $46.2B — beat vs. $45.47B expected, described as 'a record September quarter'; "
            "(4) Services $24.97B — miss vs. $25.28B expected."
        ),
        "citation": "Source: cnbc_earnings_highlight.mp4  |  timestamp_start: 0.0s  |  timestamp_end: 30.0s",
        "citation_type": "mp4",
    },
    {
        "q_num": 23, "source_file": "cnbc_earnings_highlight.mp4", "q_type": "RAG",
        "modality": "MP4",
        "query": "What did the analyst or anchor say about the earnings beat or miss in the CNBC highlight?",
        "expected_answer": (
            "Reporter Brenda noted beats on top and bottom lines; iPhone set a record September quarter. "
            "The one miss: Services ($24.97B vs. $25.28B expected). Analyst commentary noted iPhone had "
            "showed 'no growth to low-single-digit growth' over the iPhone 14/15 cycles, and now seeing "
            "'mid-single-digit growth.' Analyst Mike Santoli observed Apple trades at ~30x PE (vs. 15x "
            "five years ago), justified by less hit-driven, more consistent Services revenue — but "
            "'a relatively demanding standard' for the stock's current valuation."
        ),
        "citation": "Source: cnbc_earnings_highlight.mp4  |  timestamp_start: 0.0s  |  timestamp_end: 108.4s",
        "citation_type": "mp4",
    },
    {
        "q_num": 24, "source_file": "cnbc_earnings_highlight.mp4", "q_type": "RAG",
        "modality": "MP4",
        "query": "What was the stock price reaction or after-hours move mentioned in the CNBC earnings video?",
        "expected_answer": (
            "Shares were 'pretty much flat' and 'fractionally moving around' right after the report. "
            "Anchor John noted 'there's more to learn on the call' before stocks settle. The analyst "
            "noted Apple's stock went from ~$195 to ~$225 after the WWDC AI capabilities unveiling and "
            "'more or less has been there for four to six months.' The record iPhone quarter was seen "
            "as 'a sigh of relief' for investors who had feared a miss."
        ),
        "citation": "Source: cnbc_earnings_highlight.mp4  |  timestamp_start: 108.4s  |  timestamp_end: 132.8s",
        "citation_type": "mp4",
    },
    {
        "q_num": 25, "source_file": "Web Search", "q_type": "WEB",
        "modality": "WEB",
        "query": "What are the most recent/latest S&P 500 Q1 2026 earnings surprises for major companies? Search the internet for current earnings surprise data.",
        "expected_answer": (
            "Q1 2026 earnings season: 84% of S&P 500 companies beat EPS estimates (vs. 5-yr avg 78%; "
            "10-yr avg 76%) — the highest beat rate since Q2 2021 (87%). Earnings 20.7% above estimates "
            "on aggregate (vs. 5-yr avg 7.3%), the highest magnitude since Q1 2021 (22.2%). S&P 500 EPS "
            "growth tracking ~28% YoY. Biggest contributors: Alphabet, Amazon, and Meta Platforms "
            "(three 'Magnificent 7' names). Revenue beat rate: 81% (vs. 5-yr avg 70%). Companies that "
            "missed saw avg stock decline of ~3.9%."
        ),
        "citation": "Source: Web Search  |  Locator: internet retrieval (no local source file)",
        "citation_type": "web",
    },

    # ── File 6: fred_sp500.xlsx ─────────────────────────────────────────────
    {
        "q_num": 26, "source_file": "fred_sp500.xlsx", "q_type": "RAG",
        "modality": "XLSX",
        "query": "What was the S&P 500 closing value at the end of calendar year 2022?",
        "expected_answer": (
            "The S&P 500 closed at 3,839.50 on December 30, 2022 (the last trading day of 2022). This "
            "represented a decline of approximately 19.44% from the December 31, 2021 close of 4,766.18, "
            "making 2022 one of the worst annual performances in recent history due to aggressive Fed "
            "rate hikes and inflation concerns."
        ),
        "citation": "Source: fred_sp500.xlsx  |  sheet_name: \"S&P 500\"",
        "citation_type": "xlsx",
    },
    {
        "q_num": 27, "source_file": "fred_sp500.xlsx", "q_type": "RAG",
        "modality": "XLSX",
        "query": "What was the highest S&P 500 value recorded in the dataset, and on what date?",
        "expected_answer": (
            "The highest S&P 500 closing value in the dataset is 4,796.56, recorded on January 3, 2022. "
            "This was the all-time high at the start of 2022, just before the bear market driven by "
            "Federal Reserve tightening. The dataset spans 2021-01-04 to 2023-12-29 (753 trading days), "
            "with the lowest close of 3,577.03 on October 12, 2022."
        ),
        "citation": "Source: fred_sp500.xlsx  |  sheet_name: \"S&P 500\"",
        "citation_type": "xlsx",
    },
    {
        "q_num": 28, "source_file": "fred_sp500.xlsx", "q_type": "RAG",
        "modality": "XLSX",
        "query": "What was the percentage change in the S&P 500 from the start of 2021 to the end of 2022?",
        "expected_answer": (
            "The S&P 500 changed +3.75% from the start of 2021 to the end of 2022: from 3,700.65 "
            "(January 4, 2021 — first trading day of 2021) to 3,839.50 (December 30, 2022). This masks "
            "a 2021 rally (+27.2%, to 4,766.18) followed by a 2022 bear market (−19.44%). Over the "
            "two-year span the net change was a modest +3.75%."
        ),
        "citation": "Source: fred_sp500.xlsx  |  sheet_name: \"S&P 500\"",
        "citation_type": "xlsx",
    },
    {
        "q_num": 29, "source_file": "fred_sp500.xlsx", "q_type": "RAG",
        "modality": "XLSX",
        "query": "What was the average S&P 500 closing value for calendar year 2023 in the dataset?",
        "expected_answer": (
            "The average S&P 500 daily closing value for calendar year 2023 was 4,283.73. The index "
            "recovered strongly throughout 2023, starting near 3,853 (January 3, 2023) and finishing "
            "at 4,769.83 (December 29, 2023), a gain of +24.23% for the year. The dataset contains "
            "all 250 trading days of 2023."
        ),
        "citation": "Source: fred_sp500.xlsx  |  sheet_name: \"S&P 500\"",
        "citation_type": "xlsx",
    },
    {
        "q_num": 30, "source_file": "Web Search", "q_type": "WEB",
        "modality": "WEB",
        "query": "What is the S&P 500's latest/most recent current level, year-to-date return, and forward P/E ratio in 2026? Search the internet for current data.",
        "expected_answer": (
            "As of June 2026: forward 12-month P/E ≈ 21.1 (above 5-yr avg 19.9; 10-yr avg 19.0). "
            "Trailing P/E ≈ 25.4–31.8 (source-dependent). Goldman Sachs projected a 12% total return "
            "for full-year 2026. Q1 2026 earnings season showed ~28% YoY earnings growth with 84% beat "
            "rate. Recent S&P 500 level approximately in the 5,900–6,200 range based on available data "
            "(specific real-time level subject to market movement)."
        ),
        "citation": "Source: Web Search  |  Locator: internet retrieval (no local source file)",
        "citation_type": "web",
    },

    # ── File 7: gdp_unemployment.jpg ───────────────────────────────────────
    {
        "q_num": 31, "source_file": "gdp_unemployment.jpg", "q_type": "RAG",
        "modality": "JPG",
        "query": "What does the chart show about the relationship between GDP growth and unemployment rate?",
        "expected_answer": (
            "The chart shows an inverse (countercyclical) relationship: when GDP growth rises, "
            "unemployment falls, and vice versa. Notable patterns: (1) GFC 2008–2009 — GDP contracted, "
            "unemployment surged to ~10%; (2) 2010–2019 recovery — GDP grew steadily, unemployment fell "
            "to record lows; (3) COVID-19 2020 — sharpest GDP collapse with a brief unemployment spike, "
            "followed by the fastest recovery on record."
        ),
        "citation": "Source: gdp_unemployment.jpg  |  caption: \"U.S. GDP vs. Unemployment Rate 2003-2023\"",
        "citation_type": "jpg",
    },
    {
        "q_num": 32, "source_file": "gdp_unemployment.jpg", "q_type": "RAG",
        "modality": "JPG",
        "query": "What time period does the GDP and unemployment chart cover?",
        "expected_answer": (
            "The chart covers 2003 to 2023 — a span of exactly 20 years of U.S. macroeconomic data, "
            "encompassing the pre-GFC expansion, the Global Financial Crisis, the post-GFC recovery, "
            "the COVID-19 shock, and the subsequent rebound."
        ),
        "citation": "Source: gdp_unemployment.jpg  |  caption: \"U.S. GDP vs. Unemployment Rate 2003-2023\"",
        "citation_type": "jpg",
    },
    {
        "q_num": 33, "source_file": "gdp_unemployment.jpg", "q_type": "RAG",
        "modality": "JPG",
        "query": "What was the peak unemployment rate visible in the GDP unemployment chart?",
        "expected_answer": (
            "The peak unemployment rate visible in the chart is approximately 10%, occurring around "
            "2009–2010 in the aftermath of the Global Financial Crisis. There is also a steep but brief "
            "spike in 2020 (COVID-19 pandemic), though the GFC peak (~10%) appears as the highest "
            "sustained unemployment level in the 2003–2023 time window."
        ),
        "citation": "Source: gdp_unemployment.jpg  |  caption: \"U.S. GDP vs. Unemployment Rate 2003-2023\"",
        "citation_type": "jpg",
    },
    {
        "q_num": 34, "source_file": "gdp_unemployment.jpg", "q_type": "RAG",
        "modality": "JPG",
        "query": "During which period did GDP show the sharpest decline according to the chart?",
        "expected_answer": (
            "The sharpest GDP decline visible in the chart occurred in 2020 during the COVID-19 pandemic "
            "(approximately Q1–Q2 2020). This dip is visually steeper and more abrupt than the "
            "2008–2009 Great Recession contraction, though the GFC decline was more prolonged. The "
            "COVID-19 GDP collapse was followed by the fastest rebound in the chart's 20-year window."
        ),
        "citation": "Source: gdp_unemployment.jpg  |  caption: \"U.S. GDP vs. Unemployment Rate 2003-2023\"",
        "citation_type": "jpg",
    },
    {
        "q_num": 35, "source_file": "Web Search", "q_type": "WEB",
        "modality": "WEB",
        "query": "What is the latest/most recent US GDP growth rate and unemployment rate reported by the Bureau of Labor Statistics in 2026? Search the internet.",
        "expected_answer": (
            "As of May 2026 (latest BLS data): US unemployment rate is 4.3% (down from 4.4% in "
            "February 2026). GDP growth in 2026 is positive, boosted by the 2025 reconciliation act "
            "and the rebound from the FY2025 discretionary appropriations lapse, partially offset by "
            "tariff impacts and immigration enforcement effects. Average monthly private payroll growth "
            "in Q1 2026 exceeded 2.5× the 2025 monthly average, indicating strong labour market "
            "momentum."
        ),
        "citation": "Source: Web Search  |  Locator: internet retrieval (no local source file)",
        "citation_type": "web",
    },
]

# ── PDF builder ─────────────────────────────────────────────────────────────

def build_pdf(entries: list[dict], output: str) -> None:
    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2.5 * cm, bottomMargin=2 * cm,
        title="RAG Evaluation — 35 Queries",
        author="Multimodal RAG Assistant",
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Title"],
        fontSize=20, textColor=C_NAVY, spaceAfter=6,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle", parent=styles["Normal"],
        fontSize=11, textColor=C_LABEL, spaceAfter=2,
        alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "SectionHead", parent=styles["Normal"],
        fontSize=13, textColor=C_NAVY, spaceBefore=14, spaceAfter=4,
        fontName="Helvetica-Bold",
    )
    label_style = ParagraphStyle(
        "Label", parent=styles["Normal"],
        fontSize=8, textColor=C_LABEL, spaceBefore=0, spaceAfter=1,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=9, leading=13, spaceAfter=4,
        alignment=TA_JUSTIFY,
    )
    citation_style = ParagraphStyle(
        "Citation", parent=styles["Normal"],
        fontSize=8, leading=11, textColor=colors.HexColor("#444444"),
        fontName="Helvetica-Oblique",
    )
    warn_style = ParagraphStyle(
        "Warn", parent=body_style,
        textColor=C_WARN,
    )
    note_style = ParagraphStyle(
        "Note", parent=styles["Normal"],
        fontSize=8, leading=11, textColor=colors.HexColor("#555555"),
        fontName="Helvetica-Oblique", spaceAfter=4,
    )

    MODALITY_COLOUR = {
        "TXT":  colors.HexColor("#E0F2F1"),
        "DOCX": colors.HexColor("#FFF3E0"),
        "MP3":  colors.HexColor("#E8EAF6"),
        "PDF":  colors.HexColor("#FCE4EC"),
        "MP4":  colors.HexColor("#E3F2FD"),
        "XLSX": colors.HexColor("#F3E5F5"),
        "JPG":  colors.HexColor("#FFFDE7"),
        "WEB":  colors.HexColor("#E8F0FE"),
    }

    story = []

    # ── Cover page ──
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph("RAG Evaluation Document", title_style))
    story.append(Paragraph("35 Queries · Expected Answers · Citations", subtitle_style))
    story.append(Paragraph("Multimodal RAG Agentic Knowledge Assistant", subtitle_style))
    story.append(Spacer(1, 0.4 * cm))

    # Summary table
    summary_data = [
        ["Metric", "Value"],
        ["Total queries", "35"],
        ["RAG queries (from source files)", "28"],
        ["Web search queries", "7"],
        ["Source files", "7  (TXT, DOCX, MP3, PDF, MP4, XLSX, JPG)"],
        ["Target answer accuracy", "≥ 85%"],
        ["Citations included", "Yes — per-modality (page, section, timestamp, sheet, caption)"],
        ["XLSX data range", "fred_sp500.xlsx covers 2021-01-04 to 2023-12-29 (753 trading days)"],
        ["Date generated", "2026-06-11"],
    ]
    t = Table(summary_data, colWidths=[5.5 * cm, 11.5 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8F9FA")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F5F6")]),
        ("GRID", (0, 0), (-1, -1), 0.4, C_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    # Legend
    legend_data = [["Type", "Colour", "Description"]]
    legend_rows = [
        ("RAG – TXT",  "#E0F2F1",  "Flat text file — no locator"),
        ("RAG – DOCX", "#FFF3E0",  "Word document — section_title"),
        ("RAG – MP3",  "#E8EAF6",  "Audio — timestamp_start / timestamp_end"),
        ("RAG – PDF",  "#FCE4EC",  "PDF — page_number"),
        ("RAG – MP4",  "#E3F2FD",  "Video — timestamp_start / timestamp_end"),
        ("RAG – XLSX", "#F3E5F5",  "Spreadsheet — sheet_name"),
        ("RAG – JPG",  "#FFFDE7",  "Image — caption"),
        ("WEB",        "#E8F0FE",  "Web search — internet retrieval"),
    ]
    for label, hex_c, desc in legend_rows:
        legend_data.append([label, "", desc])
    t2 = Table(legend_data, colWidths=[3.5 * cm, 1.2 * cm, 12.3 * cm])
    style2 = [
        ("BACKGROUND", (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, C_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i, (_, hex_c, _) in enumerate(legend_rows, 1):
        style2.append(("BACKGROUND", (1, i), (1, i), colors.HexColor(hex_c)))
    t2.setStyle(TableStyle(style2))
    story.append(t2)
    story.append(PageBreak())

    # ── Query entries ──
    SOURCE_GROUPS = [
        ("aapl_10k_2023.txt", "File 1 — aapl_10k_2023.txt  (TXT · Apple 10-K FY2023)"),
        ("aapl_def14a_2023.docx", "File 2 — aapl_def14a_2023.docx  (DOCX · Apple Proxy 2023)"),
        ("apple_q1_fy2025_earnings_commentary.mp3", "File 3 — apple_q1_fy2025_earnings_commentary.mp3  (MP3 · Apple Q1 FY2025 Earnings Commentary)"),
        ("berkshire_letter_2022.pdf", "File 4 — berkshire_letter_2022.pdf  (PDF · Berkshire 2022 Annual Letter)"),
        ("cnbc_earnings_highlight.mp4", "File 5 — cnbc_earnings_highlight.mp4  (MP4 · CNBC Earnings Highlight)"),
        ("fred_sp500.xlsx", "File 6 — fred_sp500.xlsx  (XLSX · S&P 500 Index Data May 2025–May 2026)"),
        ("gdp_unemployment.jpg", "File 7 — gdp_unemployment.jpg  (JPG · U.S. GDP vs. Unemployment 2003-2023)"),
    ]

    for src_file, section_label in SOURCE_GROUPS:
        story.append(Paragraph(section_label, section_style))
        story.append(HRFlowable(width="100%", thickness=1, color=C_NAVY, spaceAfter=6))

        group_entries = [e for e in entries if e["source_file"] == src_file]
        # Also add the web entry after the 4 RAG entries for this file
        # (web entries are interspersed Q5, Q10, Q15, Q20, Q25, Q30, Q35)
        web_q_map = {
            "aapl_10k_2023.txt": 5,
            "aapl_def14a_2023.docx": 10,
            "apple_q1_fy2025_earnings_commentary.mp3": 15,
            "berkshire_letter_2022.pdf": 20,
            "cnbc_earnings_highlight.mp4": 25,
            "fred_sp500.xlsx": 30,
            "gdp_unemployment.jpg": 35,
        }
        web_q_num = web_q_map[src_file]
        all_group = group_entries + [e for e in entries if e["q_num"] == web_q_num]
        all_group.sort(key=lambda x: x["q_num"])

        for entry in all_group:
            q_type = entry["q_type"]
            mod = entry["modality"]
            bg_color = MODALITY_COLOUR.get(mod, colors.white)
            is_web = q_type == "WEB"

            badge = f"Q{entry['q_num']}  ·  {q_type}  ·  {mod}"
            badge_color = C_WEB if is_web else C_ACCENT

            badge_style = ParagraphStyle(
                f"Badge_{entry['q_num']}", parent=styles["Normal"],
                fontSize=9, fontName="Helvetica-Bold",
                textColor=colors.white, backColor=badge_color,
                leftIndent=4, rightIndent=4, spaceBefore=6, spaceAfter=4,
            )

            # Build inner table for the entry card
            inner_rows = [
                # Query row
                [Paragraph("QUERY", label_style),
                 Paragraph(entry["query"], body_style)],
                # Expected answer row
                [Paragraph("EXPECTED ANSWER", label_style),
                 Paragraph(entry["expected_answer"].replace("⚠", "<font color='#B71C1C'>⚠</font>"),
                           warn_style if "⚠ DATA LIMITATION" in entry["expected_answer"] else body_style)],
                # Citation row
                [Paragraph("CITATION", label_style),
                 Paragraph(entry["citation"], citation_style)],
            ]

            inner_t = Table(inner_rows, colWidths=[2.8 * cm, 13.2 * cm])
            inner_t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), bg_color),
                ("VALIGN",     (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (0, -1), 5),
                ("LEFTPADDING", (1, 0), (1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, -2), 0.3, C_BORDER),
                ("BOX", (0, 0), (-1, -1), 0.6, C_BORDER),
            ]))

            story.append(KeepTogether([
                Paragraph(badge, badge_style),
                inner_t,
                Spacer(1, 0.25 * cm),
            ]))

        story.append(Spacer(1, 0.3 * cm))

    # ── Appendix: XLSX data note ────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Appendix A — Source File Caveats & Known Limitations", section_style))
    story.append(HRFlowable(width="100%", thickness=1, color=C_NAVY, spaceAfter=6))

    caveats = [
        ("<b>aapl_def14a_2023.docx — Fiscal Year Scope (Q6)</b>",
         "The 2023 DEF 14A proxy statement (filed January 2023 for the March 2023 annual meeting) "
         "reports executive compensation for Apple's most recently completed fiscal year, which is "
         "FY2022 (ending September 2022). Q6 is aligned to this scope: it asks about Tim Cook's "
         "FY2022 compensation ($99.4M) as reported in the 2023 proxy. The RAG system should return "
         "the FY2022 figure when queried against this document."),

        ("<b>apple_q1_fy2025_earnings_commentary.mp3 — Audio Content (Q11–Q14)</b>",
         "The audio file contains CNBC-style analyst commentary. Queries Q11–Q14 are scoped to "
         "information that is actually present in the audio: iPhone revenue ($69.1B, miss), Services "
         "($26.34B, +14% YoY, beat), Greater China (−11% to $18.5B), EPS ($2.40 beat), and total "
         "revenue ($124.3B slight beat). Questions are phrased to match the audio content directly "
         "rather than referencing Tim Cook quotes or CFO commentary not present in the file."),

        ("<b>berkshire_letter_2022.pdf — Narrative vs. Detailed Financials (Q17, Q19)</b>",
         "Warren Buffett's letter narrative does not include all financial detail — detailed segment "
         "P&L and exact buyback dollar amounts are in the annual report appendix (pages K-33 to K-66), "
         "which is a separate document. The letter explicitly refers readers to those pages for "
         "detailed figures. Q17 (exact repurchase amount) and Q19 (exact underwriting P/L) may "
         "receive partial answers from the narrative section only."),
    ]

    for title_text, body_text in caveats:
        story.append(Paragraph(title_text, ParagraphStyle(
            "CaveatTitle", parent=styles["Normal"],
            fontSize=10, fontName="Helvetica-Bold", textColor=C_NAVY,
            spaceBefore=10, spaceAfter=3,
        )))
        story.append(Paragraph(body_text, body_style))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Appendix B — Citation Formats by Modality", section_style))
    story.append(HRFlowable(width="100%", thickness=1, color=C_NAVY, spaceAfter=6))

    citation_table_data = [
        ["Modality", "Source Chip Field", "Example Value", "Notes"],
        ["TXT",  "none",              "—",                       "Flat file; no structural locator"],
        ["DOCX", "section_title",     "\"Executive Compensation\"", "Heading/section name"],
        ["MP3",  "timestamp_start\ntimestamp_end", "0.0s / 30.0s", "Whisper ASR chunk boundaries"],
        ["PDF",  "page_number",       "6",                        "pdfplumber page index (1-based)"],
        ["MP4",  "timestamp_start\ntimestamp_end", "0.0s / 108.4s", "FFmpeg + Whisper chunk timestamps"],
        ["XLSX", "sheet_name",        "\"S&P 500\"",              "openpyxl worksheet name"],
        ["JPG",  "caption",           "\"U.S. GDP vs. Unemployment Rate 2003-2023\"", "Vision-extracted chart title"],
        ["WEB",  "internet retrieval","—",                        "No local source file; web tool result"],
    ]
    ct = Table(citation_table_data, colWidths=[2.0*cm, 3.5*cm, 5.5*cm, 6.0*cm])
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F5F6")]),
        ("GRID", (0, 0), (-1, -1), 0.4, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(ct)

    # footer note
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph(
        "Generated by Multimodal RAG Agentic Knowledge Assistant · Phase 29 Evaluation · 2026-06-11",
        ParagraphStyle("Footer", parent=styles["Normal"],
                       fontSize=8, textColor=C_LABEL, alignment=TA_CENTER)
    ))

    doc.build(story)
    print(f"PDF written → {output}")


if __name__ == "__main__":
    build_pdf(ENTRIES, OUTPUT)
