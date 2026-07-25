#set page(
  paper: "a4",
  flipped: true,
  margin: (x: 1.5cm, y: 1.5cm),
)
#set text(
  font: "Liberation Sans",
  size: 10pt,
)

#align(center)[
  #text(size: 16pt, weight: "bold")[Benchmark Performance Comparison: Our GRPO RLVR Model vs. SOTA Models]
  #v(0.3em)
  #text(size: 11pt, fill: rgb("#555555"))[Official AraEval Evaluation Suite Results (EMNLP 2025 Standard)]
]

#v(1em)

#table(
  columns: (2.5fr, 1.2fr, 1.5fr, 1.8fr, 2.2fr, 2.2fr, 1.8fr),
  align: (left + horizon, center + horizon, center + horizon, center + horizon, center + horizon, center + horizon, center + horizon),
  stroke: (x, y) => if y == 0 { (bottom: 1.5pt + black) } else if y == 1 { (bottom: 1pt + black) } else { (bottom: 0.5pt + rgb("#e0e0e0")) },
  fill: (x, y) => if y == 0 { rgb("#f0f4f8") } else if y == 1 { rgb("#e6f3ff") } else { none },
  
  [*Model*], [*Params*], [*Access Type*], [*AraMath (Math)*], [*Ar-IFEval (Strict)*], [*AraPro (Knowledge)*], [*Macro Avg*],
  
  [*Our Model (GRPO RLVR)*], [*4B*], [*Open LoRA*], [*96.36%*], [*73.13%*], [*69.37%*], [*79.62%*],
  [Gemini 1.5 Pro], [-], [Closed API], [94.88%], [74.81%], [76.22%], [81.97%],
  [GPT-4o (gpt-4o-900ptu)], [-], [Closed API], [83.47%], [70.90%], [80.86%], [78.41%],
  [Qwen2.5-72B-Instruct], [72B], [Open Weights], [92.89%], [67.72%], [74.69%], [78.43%],
  [Claude 3.5 Sonnet], [-], [Closed API], [79.83%], [53.73%], [81.46%], [71.67%],
  [Llama-3.3-70B-Instruct], [70B], [Open Weights], [70.91%], [70.90%], [70.49%], [70.77%],
  [ALLaM-7B-Instruct-preview], [7B], [Open Weights], [66.78%], [67.65%], [69.71%], [68.05%],
  [AceGPT-v2-32B-Chat], [32B], [Open Weights], [64.46%], [63.41%], [67.19%], [65.02%],
  [Base Qwen3.5-4B], [4B], [Open Weights], [68.93%], [25.40%], [66.09%], [53.47%],
)
