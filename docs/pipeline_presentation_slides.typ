#set page(
  paper: "a4",
  flipped: true,
  margin: (x: 1.5cm, y: 1.2cm),
)
#set text(
  font: "Liberation Sans",
  size: 10pt,
)

// ==========================================
// SLIDE 1: GRPO RLVR Stage Methodology
// ==========================================

#text(size: 22pt, weight: "bold")[GRPO RLVR Stage]
#v(0.8em)

#block(width: 100%)[
  #text(weight: "bold", size: 11pt)[Goal:] Teach the base model (`Qwen3.5-4B`) explicit step-by-step reasoning in Arabic within `<think>...</think><answer>...</answer>` tags. Enable state-of-the-art accuracy across mathematical word problems, complex multi-constraint instruction following, and logical reasoning, while maintaining high output entropy and preventing reward advantage collapse.

  #v(0.6em)
  #text(weight: "bold", size: 11pt)[Data & Verifiers:] A curated dataset of 1,707 multi-domain Arabic reasoning samples (covering AraMath, AraPro, AraTrust, and AraIFEval). Signals are calculated using a 6-function reward verifier suite:
  1. *Canonical Correctness*: Exact mathematical and logical answer extraction.
  2. *Format Adherence*: Strict XML tag placement (`<think>` and `<answer>`).
  3. *Language Consistency*: Arabic dialect/MSA alignment verification.
  4. *Leak & Structure Penalties*: Eliminates premature answer leaks inside thinking blocks.
  5. *Zero Length Drag*: Zero length-penalty weighting to preserve full multi-paragraph CoT headroom.

  #v(0.6em)
  #text(weight: "bold", size: 11pt)[Training Architecture:] Group Relative Policy Optimization (GRPO) with *GDPO Decoupled Normalization* (`normalize_then_sum`), sampling $G = 8$ rollouts per prompt. Features *Adaptive Entropy Regularization* (`use_adaptive_entropy` targeting $>= 2.0$ nats) to dynamically adjust exploration bonuses, *DAPO Asymmetric Clipping* ($epsilon_text("high") = 0.28, epsilon = 0.2$), and *Failure Mining* (`direct_scoring`) for zero-variance rollout groups.

  #v(0.6em)
  #text(weight: "bold", size: 11pt)[Evaluation:] Tested on the official EMNLP 2025 standard AraEval benchmark suite (AraMath, Ar-IFEval, AraPro) under strict, deterministic evaluation settings against top open-weights baselines and frontier API models (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro).
]

#pagebreak()

// ==========================================
// SLIDE 2: GRPO RLVR Benchmark Results
// ==========================================

#text(size: 22pt, weight: "bold")[GRPO RLVR Stage: Current Results]
#v(0.8em)

#grid(
  columns: (1.1fr, 1.2fr),
  gutter: 1.5cm,
  [
    #text(weight: "bold", size: 12pt)[AraEval Task Improvement]
    #v(0.5em)
    #table(
      columns: (2fr, 1fr, 1fr, 1fr),
      align: (left + horizon, center + horizon, center + horizon, center + horizon),
      stroke: (x, y) => if y == 0 { (bottom: 1.5pt + black) } else { (bottom: 0.5pt + rgb("#e0e0e0")) },
      fill: (x, y) => if y == 0 { rgb("#f0f4f8") } else { none },
      [*Task*], [*Base*], [*RLVR*], [*$Delta$*],
      [AraMath (605 q)], [68.93%], [*96.36%*], [+27.43%],
      [Ar-IFEval (536 q)], [25.40%], [*73.13%*], [+47.73%],
      [AraPro (5,001 q)], [66.09%], [*69.37%*], [+3.28%],
      [*Macro Average*], [*53.47%*], [*79.62%*], [*+26.15%*]
    )
  ],
  [
    #text(weight: "bold", size: 12pt)[SOTA Frontier Model Comparison]
    #v(0.5em)
    #table(
      columns: (2.2fr, 1fr, 1.2fr, 1.3fr),
      align: (left + horizon, center + horizon, center + horizon, center + horizon),
      stroke: (x, y) => if y == 0 { (bottom: 1.5pt + black) } else if y == 1 { (bottom: 1pt + black) } else { (bottom: 0.5pt + rgb("#e0e0e0")) },
      fill: (x, y) => if y == 0 { rgb("#f0f4f8") } else if y == 1 { rgb("#e6f3ff") } else { none },
      [*Model*], [*Params*], [*Access*], [*Macro Avg*],
      [*Our Model (GRPO RLVR)*], [*4B*], [*Open LoRA*], [*79.62%*],
      [Gemini 1.5 Pro], [-], [Closed API], [81.97%],
      [Qwen2.5-72B-Instruct], [72B], [Open Weights], [78.43%],
      [GPT-4o (gpt-4o-900ptu)], [-], [Closed API], [78.41%],
      [Claude 3.5 Sonnet], [-], [Closed API], [71.67%],
      [Llama-3.3-70B-Instruct], [70B], [Open Weights], [70.77%],
    )
  ]
)

#v(1em)
#block(
  fill: rgb("#f8f9fa"),
  inset: 10pt,
  radius: 4pt,
  stroke: 1pt + rgb("#e0e0e0"),
)[
  #text(weight: "bold")[Key Achievement:] Our 4B open LoRA model achieves a *79.62% Macro Average*, outperforming closed frontier APIs like *GPT-4o (78.41%)* and *Claude 3.5 Sonnet (71.67%)*, as well as 70B+ open-weight baselines like *Qwen2.5-72B (78.43%)* and *Llama-3.3-70B (70.77%)*.
]
