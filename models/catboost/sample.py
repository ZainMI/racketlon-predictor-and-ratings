from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from catboost_fin import PredictorPackage

OUTPUT_PATH = Path("poster/poster_images/sample_prediction.png")

predictor = PredictorPackage.load(
    "finished_models/catboost/artifacts/predictor_package"
)

player1 = "zain magdon-ismail"
player2 = "patrick moran"


def build_prediction_lines(result: dict):
    p1 = result["player1"]
    p2 = result["player2"]

    lines = []
    lines.append(("section", "MATCH PREDICTION"))
    lines.append(("title", f"{p1.title()} vs {p2.title()}"))

    for sport in ["TT", "BD", "SQ", "TN"]:
        s = result["sports"][sport]
        diff = s["pred_diff"]
        lines.append(
            (
                "body",
                f"{sport}: {s['score_p1']:>2} - {s['score_p2']:<2}   (diff {diff:+.1f})",
            )
        )

    lines.append(("section", "TOTAL"))
    lines.append(("body", f"{p1.title():<22}: {result['total_p1']}"))
    lines.append(("body", f"{p2.title():<22}: {result['total_p2']}"))
    lines.append(("body", "-" * 32))
    lines.append(("body", f"Total Diff: {result['total_diff']:+d}"))
    lines.append(("winner", f"Predicted Winner: {result['winner'].title()}"))

    return lines


def get_player_rating_lines(predictor, player: str):
    p = player.strip().lower()
    state = predictor.inference_state["player_states_by_name"].get(p)

    if state is None:
        return [("body", f"No state found for {player}")]

    lines = [("title", f"{player.title()}")]

    for sport in ["TT", "BD", "SQ", "TN"]:
        rating = state.get(f"{sport}_rating_p1", 0.0)
        pred_diff = state.get(f"{sport}_pred_diff", 0.0)
        games = int(state.get(f"{sport}_games_p1", 0))
        lines.append(
            (
                "body",
                f"{sport}: rating {rating:>6.2f}   pred {pred_diff:+5.1f}   games {games}",
            )
        )
    return lines


def render_block(draw, box, header, lines, fonts, colors):
    x1, y1, x2, y2 = box
    accent = colors["accent"]
    fg = colors["fg"]
    subfg = colors["subfg"]
    white = colors["white"]

    # outer card
    draw.rounded_rectangle(
        box, radius=28, fill="white", outline=accent, width=5
    )

    # top header bar
    header_h = 70
    draw.rounded_rectangle((x1, y1, x2, y1 + header_h), radius=28, fill=accent)
    draw.rectangle((x1, y1 + 28, x2, y1 + header_h), fill=accent)
    draw.text((x1 + 24, y1 + 16), header, fill=white, font=fonts["header"])

    y = y1 + header_h + 24
    for kind, text in lines:
        if kind == "section":
            draw.text((x1 + 24, y), text, fill=accent, font=fonts["section"])
            y += 52
        elif kind == "title":
            draw.text((x1 + 24, y), text, fill=fg, font=fonts["title"])
            y += 54
        elif kind == "winner":
            draw.text((x1 + 24, y), text, fill=accent, font=fonts["winner"])
            y += 52
        else:
            draw.text((x1 + 24, y), text, fill=subfg, font=fonts["body"])
            y += 40


def render_text_image(
    lines_left, lines_right_top, lines_right_bottom, output_path: Path
):
    width = 1800
    height = 1200

    colors = {
        "bg": (248, 250, 252),
        "fg": (20, 20, 20),
        "subfg": (35, 35, 35),
        "accent": (20, 72, 140),
        "white": (255, 255, 255),
    }

    img = Image.new("RGB", (width, height), colors["bg"])
    draw = ImageDraw.Draw(img)

    try:
        fonts = {
            "page_title": ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 52
            ),
            "header": ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 30
            ),
            "section": ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28
            ),
            "title": ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 30
            ),
            "winner": ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 30
            ),
            "body": ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial.ttf", 28
            ),
        }
    except Exception:
        default = ImageFont.load_default()
        fonts = {
            "page_title": default,
            "header": default,
            "section": default,
            "title": default,
            "winner": default,
            "body": default,
        }

    # page title
    draw.text(
        (50, 24),
        "Sample Model Output",
        fill=colors["accent"],
        font=fonts["page_title"],
    )

    # boxes
    left_box = (35, 100, 930, 1160)
    top_right_box = (965, 100, 1765, 600)
    bottom_right_box = (965, 660, 1765, 1160)

    render_block(
        draw, left_box, "Prediction Summary", lines_left, fonts, colors
    )
    render_block(
        draw, top_right_box, "Player 1 Ratings", lines_right_top, fonts, colors
    )
    render_block(
        draw,
        bottom_right_box,
        "Player 2 Ratings",
        lines_right_bottom,
        fonts,
        colors,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    result = predictor.predict_pair(player1, player2)

    prediction_lines = build_prediction_lines(result)
    p1_rating_lines = get_player_rating_lines(predictor, player1)
    p2_rating_lines = get_player_rating_lines(predictor, player2)

    render_text_image(
        prediction_lines,
        p1_rating_lines,
        p2_rating_lines,
        OUTPUT_PATH,
    )
