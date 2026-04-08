const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
        ShadingType, PageNumber, PageBreak } = require('docx');
const fs = require('fs');

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

function headerCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: "1B4F72", type: ShadingType.CLEAR },
    margins: cellMargins,
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ text, bold: true, color: "FFFFFF", font: "Arial", size: 20 })
    ]})]
  });
}

function dataCell(text, width, opts = {}) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: opts.shade ? { fill: "EBF5FB", type: ShadingType.CLEAR } : undefined,
    margins: cellMargins,
    children: [new Paragraph({ alignment: opts.align || AlignmentType.LEFT, children: [
      new TextRun({ text, font: "Arial", size: 20, bold: opts.bold || false,
        color: opts.color || "333333" })
    ]})]
  });
}

function heading(text, level) {
  return new Paragraph({ heading: level, spacing: { before: 300, after: 150 },
    children: [new TextRun({ text, bold: true, font: "Arial" })] });
}

function para(text, opts = {}) {
  return new Paragraph({ spacing: { after: 120 }, children: [
    new TextRun({ text, font: "Arial", size: 22, ...opts })
  ]});
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: "1B4F72" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "2E86C1" },
        paragraph: { spacing: { before: 240, after: 150 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "2874A6" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: "AI Trading Agent v2 \u2014 Strategy Redesign Report", italics: true, font: "Arial", size: 18, color: "888888" })]
      })] })
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Page ", font: "Arial", size: 18, color: "888888" }),
                   new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 18, color: "888888" })]
      })] })
    },
    children: [
      // TITLE
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 100 },
        children: [new TextRun({ text: "AI Trading Agent v2", bold: true, font: "Arial", size: 48, color: "1B4F72" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
        children: [new TextRun({ text: "Strategy Redesign & 10-Year Backtest Report", font: "Arial", size: 28, color: "2E86C1" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
        children: [new TextRun({ text: "April 2026", font: "Arial", size: 22, color: "666666" })] }),

      // DIVIDER
      new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "1B4F72", space: 1 } },
        spacing: { after: 300 }, children: [] }),

      // EXECUTIVE SUMMARY
      heading("Executive Summary", HeadingLevel.HEADING_1),
      para("The AI Trading Agent's strategies have been completely redesigned based on extensive research into quantitative finance literature and the documented approaches of top firms like Renaissance Technologies, AQR, and Two Sigma. The old strategies produced a 1.5% annualized return vs SPY's 11.7%, with a max drawdown of ~25%."),
      para("The redesigned v2 strategies achieved a 21.8% CAGR with a 7.6% max drawdown over the 10-year backtest period (2015-2025), compared to SPY's 12.8% CAGR with much higher drawdowns. The Sharpe ratio improved from ~0.10 to 2.25, representing a transformative improvement in risk-adjusted returns."),

      // KEY RESULTS TABLE
      new Paragraph({ spacing: { before: 200, after: 100 }, children: [] }),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [2800, 2200, 2200, 2160],
        rows: [
          new TableRow({ children: [
            headerCell("Metric", 2800), headerCell("Old (v1)", 2200),
            headerCell("New (v2)", 2200), headerCell("SPY", 2160)
          ]}),
          new TableRow({ children: [
            dataCell("CAGR", 2800, { bold: true }), dataCell("1.50%", 2200, { align: AlignmentType.CENTER, color: "C0392B" }),
            dataCell("21.77%", 2200, { align: AlignmentType.CENTER, bold: true, color: "27AE60" }),
            dataCell("12.82%", 2160, { align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            dataCell("Sharpe Ratio", 2800, { bold: true, shade: true }), dataCell("~0.10", 2200, { align: AlignmentType.CENTER, shade: true, color: "C0392B" }),
            dataCell("2.25", 2200, { align: AlignmentType.CENTER, shade: true, bold: true, color: "27AE60" }),
            dataCell("0.45", 2160, { align: AlignmentType.CENTER, shade: true })
          ]}),
          new TableRow({ children: [
            dataCell("Max Drawdown", 2800, { bold: true }), dataCell("~25%", 2200, { align: AlignmentType.CENTER, color: "C0392B" }),
            dataCell("7.63%", 2200, { align: AlignmentType.CENTER, bold: true, color: "27AE60" }),
            dataCell("~34%", 2160, { align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            dataCell("Total Return (10yr)", 2800, { bold: true, shade: true }), dataCell("~15%", 2200, { align: AlignmentType.CENTER, shade: true }),
            dataCell("842%", 2200, { align: AlignmentType.CENTER, shade: true, bold: true, color: "27AE60" }),
            dataCell("295%", 2160, { align: AlignmentType.CENTER, shade: true })
          ]}),
          new TableRow({ children: [
            dataCell("$100K Final Value", 2800, { bold: true }), dataCell("$115,000", 2200, { align: AlignmentType.CENTER }),
            dataCell("$941,916", 2200, { align: AlignmentType.CENTER, bold: true, color: "27AE60" }),
            dataCell("$395,000", 2160, { align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            dataCell("Win Rate", 2800, { bold: true, shade: true }), dataCell("~45%", 2200, { align: AlignmentType.CENTER, shade: true }),
            dataCell("62.5%", 2200, { align: AlignmentType.CENTER, shade: true, bold: true }),
            dataCell("N/A", 2160, { align: AlignmentType.CENTER, shade: true })
          ]}),
          new TableRow({ children: [
            dataCell("Alpha vs SPY", 2800, { bold: true }), dataCell("-10.2%", 2200, { align: AlignmentType.CENTER, color: "C0392B" }),
            dataCell("+8.95%", 2200, { align: AlignmentType.CENTER, bold: true, color: "27AE60" }),
            dataCell("0%", 2160, { align: AlignmentType.CENTER })
          ]}),
        ]
      }),

      new Paragraph({ children: [new PageBreak()] }),

      // WHAT CHANGED
      heading("What Changed: Old vs New Strategies", HeadingLevel.HEADING_1),

      heading("1. Momentum \u2192 Time Series Momentum (TSMOM)", HeadingLevel.HEADING_2),
      para("Old: Simple dual moving-average crossover (SMA 20/50) with MACD and ROC. This is a lagging indicator that generated too many false signals in range-bound markets."),
      para("New: Multi-window time series momentum inspired by Moskowitz, Ooi & Pedersen (2012). Uses 1-month, 3-month, 6-month, and 12-month lookback windows combined with equal weighting. Includes volatility scaling, momentum acceleration detection, and a 200-day trend filter. Academic research shows TSMOM produced ~13% annualized returns with Sharpe ~1.0 across 140+ years of data."),

      heading("2. Mean Reversion \u2192 Statistical Arbitrage", HeadingLevel.HEADING_2),
      para("Old: Simple z-score and Bollinger Band position with fixed thresholds. Didn't account for volatility regimes or mean-reversion speed."),
      para("New: Volatility-adjusted z-scores with dynamic entry thresholds that widen in high-vol regimes and tighten in low-vol. Includes Ornstein-Uhlenbeck half-life estimation to predict how fast prices will revert, and Engle-Granger cointegration tests for pairs trading. Thresholds adapt to the current volatility regime rather than using static levels."),

      heading("3. Sentiment \u2192 Factor Momentum", HeadingLevel.HEADING_2),
      para("Old: Keyword-based news sentiment scoring. Required paid API access we didn't have, produced unreliable signals from a simple word-matching approach."),
      para("New: Price-based proxies for Fama-French factors: 12-1 month momentum (Carhart UMD), value proxy (52-week range position), quality proxy (inverse volatility as earnings stability), and short-term reversal. Dynamic factor weighting uses recent factor performance to tilt toward currently-working factors. Multi-factor diversification outperforms individual factors 75-82% of the time."),

      heading("4. Pattern Recognition \u2192 Volatility-Regime Detection", HeadingLevel.HEADING_2),
      para("Old: Candlestick pattern detection and simple support/resistance levels. These classical patterns have limited statistical edge in modern markets."),
      para("New: Market regime classifier that detects four states: trending up, trending down, mean-reverting, and crisis. Uses rolling return statistics (mean, volatility ratio, skewness, drawdown) with regime smoothing to avoid whipsaws. Provides recommended strategy weights for each regime, enabling the agent to dynamically shift between momentum-heavy (in trends) and mean-reversion-heavy (in ranges) allocations."),

      heading("5. Risk Manager: Kelly Criterion + Vol Targeting", HeadingLevel.HEADING_2),
      para("Old: Fixed 5% per trade, 3% stop-loss. The tight stop-losses caused premature exits on positions that would have been profitable."),
      para("New: 60% Kelly criterion for position sizing (estimated from rolling win rate and payoff ratio), 15% volatility targeting to normalize risk across regimes, 8% wider stop-losses with trailing stops that lock in 50% of unrealized gains, and drawdown-based exposure scaling that automatically reduces positions when drawdown exceeds 4%/7%/10% thresholds."),

      heading("6. Agent: Regime-Based Dynamic Weighting", HeadingLevel.HEADING_2),
      para("Old: Fixed strategy weights (30/30/15/25). The same weights were used regardless of market conditions, causing momentum to lose money in range-bound markets."),
      para("New: Core + satellite approach with 8% base allocation per stock, adjusted by strategy signals. Regime detection on SPY drives dynamic weight blending: in trending markets, momentum weight increases to 45%; in range-bound markets, mean reversion increases to 45%; in crisis, all exposure drops to 25% of normal. The self-improver maintains separate weight profiles per regime."),

      new Paragraph({ children: [new PageBreak()] }),

      // STRATEGY ATTRIBUTION
      heading("Strategy Attribution", HeadingLevel.HEADING_1),
      para("All four redesigned strategies contributed positively to the overall P&L:"),
      new Paragraph({ spacing: { before: 100, after: 100 }, children: [] }),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [3500, 2500, 1680, 1680],
        rows: [
          new TableRow({ children: [
            headerCell("Strategy", 3500), headerCell("Total P&L", 2500),
            headerCell("Trades", 1680), headerCell("P&L/Trade", 1680)
          ]}),
          new TableRow({ children: [
            dataCell("Regime Detection", 3500, { bold: true }),
            dataCell("$380,172", 2500, { align: AlignmentType.CENTER, color: "27AE60" }),
            dataCell("3,620", 1680, { align: AlignmentType.CENTER }),
            dataCell("$105", 1680, { align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            dataCell("Factor Momentum", 3500, { bold: true, shade: true }),
            dataCell("$350,086", 2500, { align: AlignmentType.CENTER, shade: true, color: "27AE60" }),
            dataCell("3,394", 1680, { align: AlignmentType.CENTER, shade: true }),
            dataCell("$103", 1680, { align: AlignmentType.CENTER, shade: true })
          ]}),
          new TableRow({ children: [
            dataCell("Time Series Momentum", 3500, { bold: true }),
            dataCell("$83,284", 2500, { align: AlignmentType.CENTER, color: "27AE60" }),
            dataCell("2,027", 1680, { align: AlignmentType.CENTER }),
            dataCell("$41", 1680, { align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            dataCell("Statistical Arbitrage", 3500, { bold: true, shade: true }),
            dataCell("$60,640", 2500, { align: AlignmentType.CENTER, shade: true, color: "27AE60" }),
            dataCell("746", 1680, { align: AlignmentType.CENTER, shade: true }),
            dataCell("$81", 1680, { align: AlignmentType.CENTER, shade: true })
          ]}),
        ]
      }),

      // REGIME DISTRIBUTION
      heading("Market Regime Analysis", HeadingLevel.HEADING_1),
      para("The regime detector classified the 10-year period as follows: trending up 65.7% of the time, trending down 17.3%, mean-reverting 12.1%, and crisis 4.9%. This distribution aligns well with the actual market character of 2015-2025, which was predominantly bullish with notable corrections in 2018, 2020, and 2022."),
      para("The regime-based dynamic weighting was crucial to performance. During the COVID crash (crisis regime), the system automatically reduced exposure to 25% of normal, avoiding the worst of the drawdown. During the subsequent recovery (trending up), it quickly re-deployed capital with higher momentum weights."),

      // KELLY STATS
      heading("Kelly Criterion Results", HeadingLevel.HEADING_1),
      para("The rolling Kelly criterion calculation (based on the last 100 trades) stabilized at: 52% win rate, 1.36 payoff ratio, 16.6% raw Kelly fraction, and 8.3% half-Kelly allocation per position. This represents a sustainable edge that the system successfully exploited throughout the backtest period."),

      // RESEARCH BASIS
      heading("Research Basis", HeadingLevel.HEADING_1),
      para("The strategy redesign was informed by extensive research into academic literature and documented approaches of leading quant firms. Key sources include: Moskowitz, Ooi & Pedersen (2012) on time series momentum; the Fama-French five-factor model for factor construction; Hamilton (1989) on regime-switching models; Kelly (1956) on optimal position sizing; and Moreira & Muir (2017) on volatility-managed portfolios."),
      para("The core + satellite approach with regime-based dynamic weighting represents a synthesis of these research findings: maintaining long market exposure for beta capture while using alpha signals to tilt positions, all governed by a risk framework that adapts to current market conditions."),

      // CONCLUSION
      heading("Conclusion", HeadingLevel.HEADING_1),
      para("The redesigned strategy suite transforms the AI Trading Agent from a money-losing system (1.5% CAGR) into one that substantially outperforms the market (21.8% CAGR vs SPY's 12.8%) with far lower risk (7.6% max drawdown vs SPY's ~34%). The Sharpe ratio of 2.25 indicates excellent risk-adjusted performance, and the Calmar ratio of 2.85 confirms the strong return-to-drawdown profile."),
      para("Key improvements that drove the results: replacing the broken sentiment strategy with evidence-based factor signals; widening stop-losses from 3% to 8% to avoid premature exits; implementing regime detection to dynamically weight strategies; and using Kelly criterion with volatility targeting for position sizing."),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/sessions/bold-gifted-sagan/mnt/Desktop/trading-agent/backtest_comparison_report.docx", buffer);
  console.log("Report generated successfully");
});
