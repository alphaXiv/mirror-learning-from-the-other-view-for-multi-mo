import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # MIRROR on public paired geometry views

    A vision-language model can see the same geometry problem as words, a diagram,
    or both—and answer differently. MIRROR tries to turn that weakness into a
    teaching signal by letting the model's strongest view guide its weaker views.

    **Verdict: partially reproduced.** The baseline disagreed across views on
    **62.5%** of held-out problems, clearly confirming the diagnosis. Default
    reciprocal training did not reliably beat matched supervision across three
    seeds; a stronger reciprocal weight helped at seed 17 and tied the matched
    control at seed 29.

    This notebook embeds the completed evidence. It does **not** rerun the expensive
    model experiments and works directly in Molab without repository-relative data.
    """)
    return


@app.cell
def _():
    baseline = {
        "Text accuracy": 16.67,
        "Diagram accuracy": 14.58,
        "Combined accuracy": 21.88,
        "Any view correct": 34.38,
        "All views correct": 5.21,
        "Disagreement": 62.50,
    }
    seed_rows = [
        {"seed": 17, "Matched SFT": 44.10, "Adaptive λ=.01": 45.14, "No reciprocal": 45.49},
        {"seed": 29, "Matched SFT": 41.32, "Adaptive λ=.01": 40.28, "No reciprocal": 41.32},
        {"seed": 41, "Matched SFT": 30.90, "Adaptive λ=.01": 31.25, "No reciprocal": 30.90},
    ]
    kl_rows = [
        {"λ": 0.0, "Mean accuracy": 45.49, "All views correct": 34.38},
        {"λ": 0.001, "Mean accuracy": 43.40, "All views correct": 33.33},
        {"λ": 0.005, "Mean accuracy": 45.49, "All views correct": 33.33},
        {"λ": 0.01, "Mean accuracy": 45.14, "All views correct": 34.38},
        {"λ": 0.05, "Mean accuracy": 46.53, "All views correct": 37.50},
    ]
    return baseline, kl_rows, seed_rows


@app.cell
def _(mo, seed_rows):
    def grouped_svg(rows):
        colors = ["#2F6FED", "#1B9E77", "#E07A2D"]
        keys = ["Matched SFT", "Adaptive λ=.01", "No reciprocal"]
        parts = [
            '<svg viewBox="0 0 840 430" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Mean accuracy by seed">',
            '<rect width="840" height="430" fill="white"/>',
            '<text x="35" y="35" font-family="system-ui" font-size="21" font-weight="700" fill="#18202A">The default MIRROR advantage changes sign</text>',
            '<text x="35" y="59" font-family="system-ui" font-size="13" fill="#637083">Mean held-out accuracy across text, diagram, and combined views</text>',
        ]
        left, top, height = 70, 100, 250
        for tick in range(0, 61, 10):
            y = top + height - height * tick / 60
            parts.append(f'<line x1="{left}" y1="{y}" x2="815" y2="{y}" stroke="#E1E6EC"/>')
            parts.append(f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-family="system-ui" font-size="11" fill="#637083">{tick}%</text>')
        for gi, row in enumerate(rows):
            center = 175 + gi * 245
            for si, key in enumerate(keys):
                value = row[key]
                x = center + (si - 1) * 54 - 20
                h = height * value / 60
                y = top + height - h
                parts.append(f'<rect x="{x}" y="{y}" width="40" height="{h}" rx="4" fill="{colors[si]}"/>')
                parts.append(f'<text x="{x+20}" y="{y-6}" text-anchor="middle" font-family="system-ui" font-size="10" fill="#18202A">{value:.1f}</text>')
            parts.append(f'<text x="{center}" y="377" text-anchor="middle" font-family="system-ui" font-size="13" fill="#18202A">Seed {row["seed"]}</text>')
        for i, key in enumerate(keys):
            x = 115 + i * 210
            parts.append(f'<rect x="{x}" y="398" width="12" height="12" rx="2" fill="{colors[i]}"/>')
            parts.append(f'<text x="{x+18}" y="408" font-family="system-ui" font-size="11" fill="#18202A">{key}</text>')
        parts.append("</svg>")
        return "".join(parts)

    mo.Html(grouped_svg(seed_rows))
    return


@app.cell
def _(baseline, mo):
    mo.md(
        f"""
        ## 1. Does the same problem produce different behavior?

        Yes. The base model's per-view exact-answer accuracy was low, but choosing the
        successful view after the fact would reach **{baseline["Any view correct"]:.2f}%**,
        versus **{baseline["All views correct"]:.2f}%** when all three views had to be
        correct. A complementary success—some but not all views correct—occurred on
        **29.17%** of problems.
        """
    )
    mo.ui.table(
        [{"measure": name, "percent": value} for name, value in baseline.items()],
        selection=None,
        pagination=False,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. What exactly was trained?

    The formal benchmark was the MIT-licensed Geometry3K/InterGPS official split:
    128 train problems and 96 held-out problems. Each problem became three paired
    inputs:

    1. **Text:** question plus released text and diagram logic forms.
    2. **Diagram:** released diagram plus question.
    3. **Combined:** diagram, question, and logic forms.

    Qwen3-VL-4B-Instruct received 16 matched LoRA optimizer updates. The ordinary
    control used answer supervision under the two restricted views. The bounded
    MIRROR approximation selected the best view by exact-answer reward and applied
    token-level reverse KL on the same gold answer trajectory.

    This is not the paper's full algorithm: supervised LoRA replaces 16-rollout
    GRPO, and a frozen base teacher replaces an online exponential-moving-average
    teacher. Those substitutions make this a mechanism probe, not a direct estimate
    of the paper's ODA-Data scores.
    """)
    return


@app.cell
def _(kl_rows, mo):
    mo.md(
        r"""
        ## 3. Does reciprocal loss isolate the improvement?

        Not at the paper's default coefficient. At λ=0.01, the three-seed mean advantage
        over SFT was only **+0.12 percentage points**, and all-view correctness was
        slightly lower. The λ=0 no-reciprocal control also matched or exceeded default
        MIRROR on several metrics.

        The sweep nevertheless found a positive regime at seed 17: λ=0.05 reached
        **46.53%** mean accuracy and **37.50%** all-view correctness, beating matched
        SFT by 2.43 and 6.25 points. At seed 29 it tied SFT on those two metrics while
        improving agreement by 2.08 points.
        """
    )
    mo.ui.table(kl_rows, selection=None, pagination=False)
    return


@app.cell
def _(mo, seed_rows):
    deltas = [
        {
            "seed": row["seed"],
            "MIRROR − SFT (points)": round(row["Adaptive λ=.01"] - row["Matched SFT"], 2),
            "MIRROR − no reciprocal (points)": round(
                row["Adaptive λ=.01"] - row["No reciprocal"], 2
            ),
        }
        for row in seed_rows
    ]
    mo.md(
        """
        ## 4. Robustness and interpretation

        Adaptive teacher selection itself was balanced at seed 17 (42 text, 42 image,
        44 combined teachers) and narrowly led all three fixed-teacher choices in mean
        accuracy. The seed table below shows why the overall verdict remains partial:
        the sign and size of the default MIRROR effect are unstable.
        """
    )
    mo.ui.table(deltas, selection=None, pagination=False)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Claim ledger

    | Claim | Assessment in this setup |
    |---|---|
    | Equivalent views expose complementary successes and disagreement | **Aligned** |
    | Reciprocal training improves over ordinary training | **Partially aligned**; tuned λ helps, default is not robust |
    | Adaptive teachers beat fixed teachers | **Partially aligned**; narrow mean lead, mixed other metrics |
    | Reverse KL isolates the mechanism | **Inconclusive**; no-loss is competitive |

    ## Compute and provenance

    Every formal result ran through OpenResearch on **Kubernetes** using
    **NVIDIA RTX PRO 6000 Blackwell** GPUs. Jobs used four GPUs each, with a peak
    of **16 concurrently allocated GPUs**. The successful instrumented evidence
    window was **0.3121 wall hours**, from 2026-07-27 02:35:27 to
    02:54:10.555 UTC. Early launcher and optional dependency compatibility attempts
    ended before scientific execution and are excluded.

    The paper reports 42.5% → 60.7% both-view solvability for MIRROR, versus 53.6%
    for standard GRPO. Those numbers are context, not a direct comparison: this
    notebook uses a different public dataset, greedy pass@1, all three views, and
    the bounded training approximation described above.
    """)
    return


if __name__ == "__main__":
    app.run()
