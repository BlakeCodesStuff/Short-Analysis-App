"""Author-selected SHORT research cases; independent of interactive screen settings.

These are written investment arguments, not generated financial-model outputs.
The rankings and warning scores refer only to the saved default BM → STA → GPA screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape


CASE_INTRO = (
    "I chose three companies as my investment cases: FTAI Aviation, Coherent, and "
    "Rocket Lab. Although these do score high on my custom SHORT ranking, you will "
    "notice that they are not the top three ranked companies in that list. While "
    "my custom short ranking does push forward the companies with the most potential "
    "financial warning signs, the custom short warning score does not necessarily "
    "determine the most convincing short investment. We must look deeper into the "
    "top 10 companies and identify the contenders with the strongest evidence "
    "supporting a sell/short decision. Below are my cases that support my choices."
)


@dataclass(frozen=True)
class InvestmentCase:
    ticker: str
    company: str
    default_rank: int
    default_score: float
    fiscal_year: str
    metrics: tuple[tuple[str, str], ...]
    body: tuple[str, ...]
    counterargument: str
    conclusion: str
    evidence: tuple[str, ...]


CASES = (
    InvestmentCase(
        ticker="FTAI",
        company="FTAI Aviation",
        default_rank=3,
        default_score=90.9,
        fiscal_year="FY 2025",
        metrics=(("BM", "0.017"), ("OCF", "−$311M"), ("Debt", "$3.45B")),
        body=(
            "The sell/short opportunity is presented due to its high market valuation "
            "paired with substantial debt obligations, and potential concern surrounding "
            "the relationship between its reported earnings and operating cash flow.",
            "At the end of the 2025 fiscal year, FTAI had a BM ratio of approximately "
            "0.017, indicating that its MV was about 60 times its reported common "
            "equity. They also carried around $3.45 billion in debt compared with "
            "only $334 million in shareholders equity. This financial structure could "
            "increase its vulnerability if the company’s operating performance weakens.",
            "One of the largest warning signs was the difference between their "
            "earnings and cash flow. In 2025, FTAI reported approximately $501 "
            "million in net income as well as negative operating cash flow of $311 "
            "million. Its STA ratio of 18.56% reinforces my point.",
        ),
        counterargument=(
            "However, it is important to be careful interpreting this discrepancy. "
            "The company’s business model makes use of aircraft leasing, engine "
            "maintenance, and asset sales, with some cash receipts being classified "
            "as investing activities instead of operating activities. As a result, "
            "negative OCF does not automatically indicate low-quality earnings. In "
            "fact, its aerospace products division continued to grow rapidly in 2026."
        ),
        conclusion=(
            "Regardless of these considerations, FTAI’s heightened valuation, "
            "significant leverage, and complex cash flow structure still do support "
            "further leaning towards a short candidate. If earnings don’t convert "
            "into sustainable cash generation, I would reconsider the premium that "
            "the company is currently selling at."
        ),
        evidence=(
            "2025 net income: $501 million",
            "2025 OCF: −$311 million",
            "2025 debt: $3.45 billion",
            "2025 shareholders equity: $334 million",
            "BM: 0.017",
            "STA: 18.56%",
            "Q2 2026 Aerospace Products revenue growth: 78%",
        ),
    ),
    InvestmentCase(
        ticker="COHR",
        company="Coherent Corp.",
        default_rank=8,
        default_score=85.6,
        fiscal_year="FY 2026",
        metrics=(("EEY", "1.37%"), ("OCF", "$79.5M"), ("Capital spending", "$1.10B")),
        body=(
            "The sell/short opportunity with Coherent comes from its relatively high "
            "market valuation, paired with concerns surrounding its OCF and the large "
            "amount of money being spent on expanding its operations.",
            "At the end of the 2026 fiscal year, Coherent reported approximately $805 "
            "million in net income, but only $79.5 million in OCF. This was a "
            "substantial decrease from the $634 million in OCF reported in 2025. "
            "This difference is one of the largest warning signs for me, as the "
            "company's reported earnings have improved significantly without a "
            "similar improvement in cash generation. Its STA ratio of 3.96% "
            "reinforces this concern.",
            "Another important factor is the company's capital expenditures, which "
            "reached approximately $1.10 billion in 2026. This means that after "
            "accounting for its capital spending, Coherent had a negative FCF of "
            "approximately $1.02 billion. Its EEY of only 1.37% also suggests that "
            "the company's valuation is relatively high compared with the earnings "
            "generated by its operations.",
        ),
        counterargument=(
            "However, I think it is important to consider why the company's OCF has "
            "decreased. Coherent explained that much of this was due to increased "
            "inventory levels needed to support future sales. The company's revenue "
            "also increased by approximately 23% in 2026, largely supported by "
            "growing demand for AI-related technology. Additionally, Coherent "
            "received approximately $2 billion through an investment from NVIDIA, "
            "providing additional funding for its expansion."
        ),
        conclusion=(
            "Regardless of these positive developments, I believe Coherent's "
            "valuation depends heavily on its ability to continue growing and "
            "eventually convert that growth into sustainable cash flow. If the "
            "company's investments and increased inventory do not produce the "
            "expected results, I would reconsider whether its current valuation "
            "is justified."
        ),
        evidence=(
            "2026 net income: $805 million",
            "2026 OCF: $79.5 million",
            "2025 OCF: $634 million",
            "2026 capital expenditures: $1.10 billion",
            "STA: 3.96%",
            "EEY: 1.37%",
            "2026 revenue growth: approximately 23%",
        ),
    ),
    InvestmentCase(
        ticker="RKLB",
        company="Rocket Lab",
        default_rank=2,
        default_score=91.4,
        fiscal_year="FY 2025",
        metrics=(("ROC", "−13.71%"), ("EEY", "−0.50%"), ("Net loss", "$198M")),
        body=(
            "The sell/short opportunity for Rocket Lab is mainly presented by its "
            "extremely high market valuation, despite the company continuing to "
            "report losses and negative OCF. I believe this makes Rocket Lab "
            "particularly vulnerable if its future performance does not meet the "
            "expectations of investors.",
            "At the end of the 2025 fiscal year, Rocket Lab had an estimated MV of "
            "approximately $41.1 billion, compared with only $602 million in annual "
            "revenue. This means that the company was valued at approximately 68 "
            "times its yearly revenue. Its EEY was also negative at approximately "
            "-0.50%, while its ROC was -13.71%. These figures suggest that investors "
            "are placing significant value on the company's expected future growth "
            "rather than its current profitability.",
            "One of the largest concerns is that Rocket Lab has not yet been able "
            "to generate consistent profits. In 2025, the company reported a net "
            "loss of approximately $198 million, alongside negative OCF of $166 "
            "million. This indicates that the company is still spending more money "
            "through its operations than it is generating, which could become a "
            "concern if its expansion plans do not produce the expected returns.",
            "Another factor I considered was the company's continued expansion. "
            "Rocket Lab is developing its Neutron launch vehicle and pursuing the "
            "acquisition of Iridium Communications. In September 2026, it raised "
            "approximately $1.94 billion by issuing 29.3 million additional shares "
            "to help finance the acquisition. Although this improved the company's "
            "ability to fund the transaction, it also diluted the ownership of "
            "existing shareholders.",
        ),
        counterargument=(
            "However, Rocket Lab has shown significant growth that could justify "
            "some of its high valuation. In the second quarter of 2026, revenue "
            "increased by 62% compared with the previous year, reaching "
            "approximately $234 million. Its backlog also reached $2.36 billion, "
            "suggesting that demand for its products and services remains strong."
        ),
        conclusion=(
            "Despite these positive developments, I believe Rocket Lab's valuation "
            "leaves little room for disappointing results. If the company struggles "
            "to turn its rapid growth into profitability, or experiences setbacks "
            "in its expansion plans, I would expect investors to reconsider the "
            "high premium they are currently placing on the company. This supports "
            "my decision to consider Rocket Lab a potential short candidate."
        ),
        evidence=(
            "2025 revenue: $602 million",
            "2025 net loss: $198 million",
            "2025 OCF: −$166 million",
            "EEY: −0.50%",
            "ROC: −13.71%",
            "Q2 2026 revenue growth: 62%",
            "Q2 2026 backlog: $2.36 billion",
        ),
    ),
)


CASE_STYLE = """
<style>
.investment-case {
    border: 1px solid #464967;
    border-radius: 16px;
    background: linear-gradient(140deg, #292a47 0%, #20243c 68%, #202b43 100%);
    color: #eafbf7;
    padding: clamp(19px, 3vw, 34px);
    margin: 10px 0 22px;
    box-shadow: 0 12px 34px rgba(5, 8, 24, 0.15);
}
.investment-case .case-eyebrow {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 8px;
    font-size: .79rem;
    font-weight: 700;
    letter-spacing: .09em;
    text-transform: uppercase;
    color: #98d7d1;
}
.investment-case .case-badge {
    border: 1px solid #605b82;
    color: #dcdbf4;
    background: rgba(137,128,201,.12);
    border-radius: 30px;
    padding: 5px 10px;
    letter-spacing: 0;
    font-size: .8rem;
    font-weight: 600;
    text-transform: none;
}
.investment-case h3 {color: #fff; margin: 0 0 22px; font-size: clamp(1.3rem, 2.1vw, 1.7rem);}
.investment-case .case-metrics {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 24px;
}
.investment-case .case-metric {
    background: #191e36;
    border: 1px solid #393e5b;
    border-radius: 11px;
    padding: 13px 16px;
}
.investment-case .case-metric-label {
    color: #b9c4dc;
    font-size: .78rem;
    margin-bottom: 4px;
}
.investment-case .case-metric-value {font-size: clamp(1.16rem, 2vw, 1.5rem); font-weight: 700;}
.investment-case .case-body p {margin: 0 0 16px; line-height: 1.75; color: #ebf1f7;}
.investment-case .case-callout {
    padding: 17px 19px;
    border-radius: 0 10px 10px 0;
    margin: 23px 0 0;
}
.investment-case .case-callout-title {
    display: block;
    font-weight: 700;
    font-size: .81rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: 9px;
}
.investment-case .case-callout p {margin: 0; line-height: 1.72;}
.investment-case .case-counterargument {
    background: rgba(77, 204, 189, .10);
    border-left: 4px solid #4dccbd;
}
.investment-case .case-counterargument .case-callout-title {color: #7ce7d8;}
.investment-case .case-takeaway {
    background: rgba(137, 128, 201, .12);
    border-left: 4px solid #a79de8;
}
.investment-case .case-takeaway .case-callout-title {color: #c4bbff;}
.investment-case .case-evidence {
    margin-top: 24px;
    border-top: 1px solid #464967;
    padding-top: 19px;
}
.investment-case .case-evidence h4 {
    color: #eafbf7;
    margin: 0 0 12px;
    font-size: 1.02rem;
}
.investment-case .case-evidence ul {
    list-style: none;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px 18px;
    padding: 0;
    margin: 0;
}
.investment-case .case-evidence li {
    position: relative;
    padding-left: 18px;
    color: #cfd9e8;
    line-height: 1.45;
    font-size: .91rem;
}
.investment-case .case-evidence li::before {
    content: "";
    position: absolute;
    left: 2px;
    top: .56em;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #ff9991;
}
@media (max-width: 620px) {
    .investment-case .case-metrics {grid-template-columns: 1fr;}
    .investment-case .case-evidence ul {grid-template-columns: 1fr;}
}
</style>
"""


def case_html(case: InvestmentCase) -> str:
    """Render a self-contained, escaped, read-only investment case card."""
    metrics = "".join(
        '<div class="case-metric">'
        f'<div class="case-metric-label">{escape(label)}</div>'
        f'<div class="case-metric-value">{escape(value)}</div>'
        '</div>'
        for label, value in case.metrics
    )
    body = "".join(f"<p>{escape(paragraph)}</p>" for paragraph in case.body)
    evidence = "".join(f"<li>{escape(item)}</li>" for item in case.evidence)
    return (
        '<article class="investment-case">'
        '<div class="case-eyebrow">'
        f'{escape(case.ticker)} · {escape(case.fiscal_year)}'
        f'<span class="case-badge">Default SHORT rank #{case.default_rank}</span>'
        f'<span class="case-badge">Warning score {case.default_score:.1f}/100</span>'
        '</div>'
        f'<h3>{escape(case.company)}: Potential SELL/SHORT Investment Case</h3>'
        f'<div class="case-metrics">{metrics}</div>'
        f'<div class="case-body">{body}</div>'
        '<div class="case-callout case-counterargument">'
        '<span class="case-callout-title">Counterargument</span>'
        f'<p>{escape(case.counterargument)}</p>'
        '</div>'
        '<div class="case-callout case-takeaway">'
        '<span class="case-callout-title">My conclusion</span>'
        f'<p>{escape(case.conclusion)}</p>'
        '</div>'
        f'<div class="case-evidence"><h4>Supporting Evidence</h4><ul>{evidence}</ul></div>'
        '</article>'
    )
