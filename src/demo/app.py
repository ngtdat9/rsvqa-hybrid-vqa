"""The Week 7 demo: a local web app on the laptop. Pick a tile, ask one of its questions, and see the
model's answer, its occlusion map, the five most similar training cases and the explanation.

    python -m src.demo.app [--config configs/hr_config.yaml] [--bundle DIR] [--index DIR] [--no-llm]

Opens at http://127.0.0.1:7860 and serves only this machine. Showcase questions come precomputed from the
bundle; other questions run live here (about 5 s on an M1). The neighbour paragraph of a live question
comes from the LLM when GROQ_API_KEY is set and the network answers, and from the template otherwise.
"""
from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # nothing leaves the laptop but LLM calls

import gradio as gr  # noqa: E402
import yaml  # noqa: E402

from src.demo.predict import Demo, DemoAnswer, heatmap_overlay  # noqa: E402

SPLIT_LABELS = {"test": "Test (same region as training)", "test_phili": "Philadelphia (unseen city)"}
TYPE_LABELS = {"presence": "presence", "comp": "comparison", "area": "area", "count": "count"}
WRITER_LABELS = {"code": "written by code", "llm": "written by the LLM, checked by code",
                 "template": "fixed template (the LLM was not used or its text failed a check)"}

CSS = """
.answer-big { font-size: 30px; font-weight: 700; margin-right: 12px; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 13px; font-weight: 600;
         vertical-align: 6px; }
.badge.warn { background: rgba(230,159,0,.18); color: #b45309; }
.badge.ok { background: rgba(0,158,115,.15); color: #1a7f37; }
.badge.bad { background: rgba(213,94,0,.15); color: #c0362c; }
.dark .badge.warn { color: #f0a24b; } .dark .badge.ok { color: #57c27a; } .dark .badge.bad { color: #f07b72; }
.meta { opacity: .75; font-size: 13px; margin-top: 4px; }
.expl { line-height: 1.75; font-size: 15px; }
.seg { padding: 1px 3px; border-radius: 3px; }
.show .seg.code { background: rgba(86,180,233,.22); }
.show .seg.llm { background: rgba(240,200,60,.28); }
.show .seg.template { background: rgba(150,150,150,.25); }
.legend { font-size: 12.5px; opacity: .8; margin-top: 8px; }
.legend i { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin: 0 5px 0 12px;
            vertical-align: -1px; }
"""


# ---------------------------------------------------------------------------
# Pure rendering helpers (tested without a browser)
# ---------------------------------------------------------------------------

def tile_items(demo: Demo, split: str) -> list[tuple[str, str]]:
    """Gallery items (image path, caption) for one test split."""
    return [(str(t["path"]), f"tile {t['img_id']}" + (" · black strip" if t["strip"] else ""))
            for t in demo.tiles() if t["split"] == split]


def question_choices(demo: Demo, img_id: int) -> list[tuple[str, int]]:
    """Dropdown choices (label, row): precomputed questions first, marked with a star."""
    return [(("★ " if q["showcase"] else "") + f"{q['question']}  ({TYPE_LABELS.get(q['type'], q['type'])})", q["row"])
            for q in demo.questions(img_id)]


def answer_html(answer: DemoAnswer) -> str:
    unit = ""
    if answer.qtype == "count":
        unit = " object" if answer.answer == "1" else " objects"
    confidence = ("no confidence for counts" if answer.confidence is None
                  else f"confidence {answer.confidence:.0%}")
    badge = ('<span class="badge warn">⚠ Uncertain: the similar cases disagree</span>' if answer.warning
             else '<span class="badge ok">No warning</span>')
    how = "precomputed on the GPU" if answer.precomputed else "computed live on this laptop" + (
        " in " + ", ".join(f"{name} {s:.1f} s" for name, s in answer.seconds.items()) if answer.seconds else "")
    return (f'<div><span class="answer-big">{html.escape(answer.answer)}{unit}</span>{badge}</div>'
            f'<div class="meta">“{html.escape(answer.question)}” · {TYPE_LABELS.get(answer.qtype, answer.qtype)} '
            f'question · {confidence}</div><div class="meta">{how}. No warning does not mean the answer is right.</div>')


def truth_markdown(answer: DemoAnswer) -> str:
    return (f"**True answer: {answer.truth}** · the model was **{'right' if answer.correct else 'wrong'}**"
            + (" · the tile has a black no-data strip" if answer.strip else ""))


def explanation_html(answer: DemoAnswer, show_writers: bool) -> str:
    spans = " ".join(f'<span class="seg {who}" title="{WRITER_LABELS[who]}">{html.escape(text)}</span>'
                     for who, text in answer.parts)
    legend = ""
    if show_writers:
        used = dict.fromkeys(who for who, _ in answer.parts)
        colours = {"code": "rgba(86,180,233,.5)", "llm": "rgba(240,200,60,.6)", "template": "rgba(150,150,150,.5)"}
        legend = '<div class="legend">' + "".join(
            f'<i style="background:{colours[w]}"></i>{WRITER_LABELS[w]}' for w in used) + "</div>"
    return f'<div class="expl{" show" if show_writers else ""}">{spans}</div>{legend}'


def neighbour_items(answer: DemoAnswer) -> list[tuple[str, str]]:
    return [(str(n.thumb), f"#{n.rank} · true answer {n.answer} · {n.question}")
            for n in answer.neighbours if n.thumb is not None]


# ---------------------------------------------------------------------------
# The UI
# ---------------------------------------------------------------------------

def build_ui(demo: Demo) -> gr.Blocks:
    splits = [s for s in SPLIT_LABELS if any(t["split"] == s for t in demo.tiles())]
    alpha = demo.config["demo_heatmap_alpha"]

    def tiles_for(split: str) -> tuple[Any, ...]:
        items = tile_items(demo, split)
        img_ids = [t["img_id"] for t in demo.tiles() if t["split"] == split]
        first = img_ids[0] if img_ids else None
        choices = question_choices(demo, first) if first is not None else []
        return (gr.update(value=items, selected_index=0 if items else None), img_ids, first,
                gr.update(choices=choices, value=choices[0][1] if choices else None),
                demo.tile(first) if first is not None else None)

    def pick_tile(img_ids: list[int], event: gr.SelectData) -> tuple[Any, ...]:
        img_id = img_ids[event.index]
        choices = question_choices(demo, img_id)
        return img_id, gr.update(choices=choices, value=choices[0][1] if choices else None), demo.tile(img_id)

    def ask(row: int | None, show_writers: bool) -> tuple[Any, ...]:
        if row is None:
            raise gr.Error("Pick a tile and a question first.")
        answer = demo.ask(int(row))
        tile = demo.tile(answer.img_id)
        return (answer_html(answer), gr.update(open=False), truth_markdown(answer), tile,
                heatmap_overlay(tile, answer.occlusion, alpha), explanation_html(answer, show_writers),
                neighbour_items(answer), answer)

    def rewrite(answer: DemoAnswer | None, show_writers: bool) -> str:
        return explanation_html(answer, show_writers) if answer is not None else ""

    with gr.Blocks(title="Aerial VQA Demo", css=CSS, theme=gr.themes.Soft(primary_hue="sky")) as ui:
        gr.Markdown("## Explainable VQA on Aerial Images\nAsk a question about an aerial tile. See the model's "
                    "answer, what the answer depended on (occlusion), and the most similar training cases.")
        answer_state = gr.State(None)
        tile_ids = gr.State([])
        current_tile = gr.State(None)
        with gr.Tab("Demo"):
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    split = gr.Radio([(SPLIT_LABELS[s], s) for s in splits], value=splits[0] if splits else None,
                                     label="1 · Test set")
                    gallery = gr.Gallery(label="Pick a tile", columns=4, height="auto", allow_preview=False,
                                         show_label=True, object_fit="cover")
                    question = gr.Dropdown(label="2 · Ask one of this tile's questions (★ = precomputed)",
                                           choices=[], interactive=True)
                    ask_button = gr.Button("Ask", variant="primary")
                with gr.Column(scale=2):
                    answer_box = gr.HTML()
                    with gr.Accordion("Show the true answer", open=False) as truth_panel:
                        truth_box = gr.Markdown()
                    with gr.Row():
                        tile_image = gr.Image(label="Tile", type="pil", interactive=False, height=360)
                        map_image = gr.Image(label="Occlusion: brighter = covering this cell changed the answer more",
                                             type="pil", interactive=False, height=360)
                    show_writers = gr.Checkbox(label="Show who wrote what", value=False)
                    explanation_box = gr.HTML()
                    neighbours = gr.Gallery(label="The 5 most similar training cases, found after the model answered",
                                            columns=5, height=260, allow_preview=True)
        with gr.Tab("About & limits"):
            gr.Markdown(ABOUT)

        ui.load(tiles_for, split, [gallery, tile_ids, current_tile, question, tile_image])
        split.change(tiles_for, split, [gallery, tile_ids, current_tile, question, tile_image])
        gallery.select(pick_tile, tile_ids, [current_tile, question, tile_image])
        outputs = [answer_box, truth_panel, truth_box, tile_image, map_image, explanation_box, neighbours,
                   answer_state]
        ask_button.click(ask, [question, show_writers], outputs)
        show_writers.change(rewrite, [answer_state, show_writers], explanation_box)
    return ui


ABOUT = """
### Key results
| | |
|---|---|
| Test accuracy | **90.2%** (count error 0.91 objects) |
| Philadelphia, a city never seen in training | **76.9%** |
| Above zero-shot CLIP (ViT-L/14), test set | **+40.8 points** |
| Answers are more likely wrong when the warning fires | **2.4–4×** (both test sets pooled) |

### How to read the demo
- **Occlusion map:** each cell was covered in turn; brighter cells changed the answer more. It shows what the
  answer depended on, not where objects are.
- **Similar cases:** found after the model answered, by comparing the model's own representation. They did not
  cause the answer. Their true answers drive the warning.
- **Explanation:** code writes the answer and the cells; the LLM writes only the paragraph about the similar
  cases, and code checks it, falling back to a fixed template.

### Limits
- The warning is rare (1–12% of questions): **no warning does not mean the answer is right.**
- Accuracy drops 13 points in a new city.
- Only zero-shot CLIP was compared, not a CLIP fine-tuned on this data.
- The explanations were not rated by people.
"""


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/hr_config.yaml"))
    parser.add_argument("--bundle", type=Path, help="default: the config's demo_bundle_dir")
    parser.add_argument("--index", type=Path, help="default: the config's demo_index_dir")
    parser.add_argument("--no-llm", action="store_true", help="template paragraphs for live questions")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    demo = Demo(args.bundle or Path(config["demo_bundle_dir"]), args.index or Path(config["demo_index_dir"]),
                config, use_llm=not args.no_llm)
    print(f"Loaded {len(demo.tiles())} tiles on {demo.device.type}; LLM for live questions: "
          f"{'on' if demo.client else 'off (template)'}")
    build_ui(demo).launch(server_name="127.0.0.1", server_port=args.port, share=False, show_api=False,
                          allowed_paths=[str(demo.bundle_dir.resolve())])  # tiles and thumbnails live there


if __name__ == "__main__":
    _main()
