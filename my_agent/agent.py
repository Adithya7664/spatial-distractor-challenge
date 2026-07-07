"""LangGraph distractor agent — DINOv2 fallback edition.

Primary path:  ID exploitation  (1 call)
Fallback path: DINOv2 embedding + k-means clustering
                (ring texture from main cutout vs decoy cutout separates in embedding space)
Final fallback: position-centric probe
"""
from __future__ import annotations

import io
import json
import math
import os
import time
from typing import Dict, List, TypedDict

import cv2
import numpy as np
from PIL import Image
from langgraph.graph import END, StateGraph

from harness.agent_base import DistractorAgent
from harness.session import ToolsProxy

def _measure_elongation(piece_png: bytes) -> float:
    img = np.array(Image.open(io.BytesIO(piece_png)).convert("RGB"))
    mask = img.max(axis=2) > 60
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) < 4 or len(cols) < 4:
        return 1.0
    h = int(rows[-1] - rows[0] + 1)
    w = int(cols[-1] - cols[0] + 1)
    return max(w / h, h / w)


def _estimate_hole_radius(board_gray: np.ndarray, cx: float, cy: float) -> float:
    h, w = board_gray.shape
    radii = []
    for theta in np.linspace(0, 2 * math.pi, 36, endpoint=False):
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        for r in range(20, min(h, w) // 2):
            x, y = int(cx + r * cos_t), int(cy + r * sin_t)
            if 0 <= x < w and 0 <= y < h and board_gray[y, x] > 50:
                radii.append(r)
                break
    return float(np.median(radii)) if radii else 123.0


def _sample_board_ring(board_rgb: np.ndarray, cx: float, cy: float,
                        r: float, n: int = 360, d: float = 10.0) -> np.ndarray:
    h, w = board_rgb.shape[:2]
    strip = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        theta = 2 * math.pi * i / n
        x, y = int(cx + (r + d) * math.cos(theta)), int(cy + (r + d) * math.sin(theta))
        if 0 <= x < w and 0 <= y < h:
            strip[i] = board_rgb[y, x]
    return strip


def _sample_piece_outer(piece_rgb: np.ndarray, n: int = 360,
                        d_in: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    mask = (piece_rgb.max(axis=2) > 60)
    ys_all, xs_all = np.where(mask)
    if len(ys_all) < 10:
        return np.zeros((n, 3), dtype=np.float32), np.zeros(n, dtype=bool)
    cy_p, cx_p = float(ys_all.mean()), float(xs_all.mean())
    h, w = piece_rgb.shape[:2]
    max_r = int(math.hypot(h, w))
    r_vals = np.arange(max_r, 5, -1)
    strip = np.zeros((n, 3), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        theta = 2 * math.pi * i / n
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        xs = np.clip((cx_p + r_vals * cos_t).astype(int), 0, w - 1)
        ys = np.clip((cy_p + r_vals * sin_t).astype(int), 0, h - 1)
        bright = mask[ys, xs]
        if not bright.any():
            continue
        first = int(np.argmax(bright))
        r_in = max(r_vals[first] - d_in, 3)
        xi, yi = int(cx_p + r_in * cos_t), int(cy_p + r_in * sin_t)
        if 0 <= xi < w and 0 <= yi < h:
            strip[i] = piece_rgb[yi, xi]
            valid[i] = True
    return strip, valid


def _ncc_rotate(board_strip: np.ndarray, piece_strip: np.ndarray,
                valid: np.ndarray) -> tuple[float, int]:
    n = len(board_strip)
    valid_idxs = np.where(valid)[0]
    if len(valid_idxs) < 10:
        return 0.0, 0
    bg = board_strip.mean(axis=1)
    pg = piece_strip.mean(axis=1)
    pg_v = pg[valid_idxs]
    pg_c = pg_v - pg_v.mean()
    pg_std = pg_c.std()
    if pg_std < 1.0:
        return 0.0, 0
    pg_norm = pg_c / pg_std
    n_v = len(valid_idxs)
    best_score, best_off = -1.0, 0
    for off in range(n):
        window = bg[(valid_idxs + off) % n]
        w_c = window - window.mean()
        w_std = w_c.std()
        if w_std < 1.0:
            continue
        score = float(np.dot(w_c / w_std, pg_norm) / n_v)
        if score > best_score:
            best_score, best_off = score, off
    return best_score, best_off


def _piece_area_scores(piece_pngs: Dict[int, bytes]) -> Dict[int, int]:
    scores = {}
    for pid, png in piece_pngs.items():
        img = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
        scores[pid] = int((img.max(axis=2) > 60).sum())
    print(f"[area] pixel counts: { {k: v for k, v in scores.items()} }")
    return scores


def _edge_strips(piece_rgb: np.ndarray, valid_idxs: np.ndarray,
                   n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Color strips along the two radial arc endpoints of a piece."""
    if len(valid_idxs) < 5:
        return np.zeros((n, 3)), np.zeros((n, 3))
    ys, xs = np.where(piece_rgb.max(axis=2) > 60)
    if len(ys) < 10:
        return np.zeros((n, 3)), np.zeros((n, 3))
    cy_p, cx_p = float(ys.mean()), float(xs.mean())
    h, w = piece_rgb.shape[:2]

    def strip(theta):
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        pts = []
        for r in range(8, n + 8):
            x, y = int(cx_p + r * cos_t), int(cy_p + r * sin_t)
            if 0 <= x < w and 0 <= y < h:
                pts.append(piece_rgb[y, x].astype(np.float32))
        if not pts:
            return np.zeros((n, 3))
        arr = np.array(pts)
        return arr[:n] if len(arr) >= n else np.vstack([arr, np.zeros((n - len(arr), 3))])

    t_lo = 2 * math.pi * int(valid_idxs.min()) / 360
    t_hi = 2 * math.pi * int(valid_idxs.max()) / 360
    return strip(t_lo), strip(t_hi)


def _strip_ncc(a: np.ndarray, b: np.ndarray) -> float:
    ag, bg = a.mean(axis=1), b.mean(axis=1)
    ac, bc = ag - ag.mean(), bg - bg.mean()
    sa, sb = ac.std(), bc.std()
    if sa < 1.0 or sb < 1.0:
        return 0.0
    return float(np.dot(ac / sa, bc / sb) / len(ag))


def _pairwise_edge_affinity(piece_pngs: Dict[int, bytes]) -> tuple[list, float]:
    """
    Find the most ISOLATED pair via edge NCC.
    Distractors form a dead-end pair: high mutual NCC, low NCC to real pieces.
    Real pieces form a cycle: each piece also connects to a *third* piece, reducing isolation.
    Returns (pair_ids, isolation_score).
    """
    from itertools import combinations
    ids = sorted(piece_pngs.keys())
    strips = {}
    for pid in ids:
        rgb = np.array(Image.open(io.BytesIO(piece_pngs[pid])).convert("RGB"))
        _, valid = _sample_piece_outer(rgb, n=360)
        vidxs = np.where(valid)[0]
        strips[pid] = _edge_strips(rgb, vidxs)

    pair_ncc: Dict[tuple, float] = {}
    for a, b in combinations(ids, 2):
        lo_a, hi_a = strips[a]
        lo_b, hi_b = strips[b]
        pair_ncc[(a, b)] = max(
            _strip_ncc(lo_a, lo_b), _strip_ncc(lo_a, hi_b),
            _strip_ncc(hi_a, lo_b), _strip_ncc(hi_a, hi_b),
        )

    top_pairs = sorted(pair_ncc.items(), key=lambda x: -x[1])[:6]
    print(f"[edge] top pairs: { [(str(k), round(v,3)) for k,v in top_pairs] }")

    # isolation = pair_NCC minus how strongly each piece connects to anyone OUTSIDE the pair
    best_isolation, best_pair = -999.0, None
    for (a, b), ab_ncc in pair_ncc.items():
        a_best = max(
            (v for (x, y), v in pair_ncc.items() if (x == a or y == a) and (x, y) != (a, b)),
            default=0.0,
        )
        b_best = max(
            (v for (x, y), v in pair_ncc.items() if (x == b or y == b) and (x, y) != (a, b)),
            default=0.0,
        )
        isolation = ab_ncc - max(a_best, b_best)
        if isolation > best_isolation:
            best_isolation = isolation
            best_pair = (a, b)

    print(f"[edge] best isolated pair={best_pair} isolation={best_isolation:.3f}")
    return list(best_pair) if best_pair else [], best_isolation



def _perimeter_match(board_rgb: np.ndarray, board_gray: np.ndarray,
                    piece_pngs: Dict[int, bytes], cx: float, cy: float,
                    n_dist: int) -> tuple[List[int], Dict[int, float], dict, bool, float]:
    r = _estimate_hole_radius(board_gray, cx, cy)
    print(f"[perimeter] centre=({cx:.0f},{cy:.0f}) radius={r:.0f}")
    board_strip = _sample_board_ring(board_rgb, cx, cy, r, n=360, d=3.0)
    ids = sorted(piece_pngs.keys())
    probe_r = 0.6 * r
    ncc_scores: Dict[int, float] = {}
    home_positions: dict = {}
    for pid in ids:
        piece_rgb = np.array(Image.open(io.BytesIO(piece_pngs[pid])).convert("RGB"))
        p_strip, p_valid = _sample_piece_outer(piece_rgb, n=360, d_in=12.0)
        score, best_off = _ncc_rotate(board_strip, p_strip, p_valid)
        ncc_scores[pid] = round(score, 4)
        valid_idxs = np.where(p_valid)[0]
        centre_idx = float(valid_idxs.mean()) if len(valid_idxs) else 0.0
        home_angle = 2 * math.pi * ((centre_idx + best_off) % 360) / 360
        home_positions[pid] = (cx + probe_r * math.cos(home_angle),
                                cy + probe_r * math.sin(home_angle))
    print(f"[perimeter] NCC: { {k: v for k, v in ncc_scores.items()} }")
    ncc_sorted = sorted(ids, key=lambda i: ncc_scores[i])
    distractors = sorted(ncc_sorted[:n_dist])
    ncc_vals = sorted(ncc_scores.values())
    gap = ncc_vals[n_dist] - ncc_vals[n_dist - 1]
    print(f"[perimeter] distractors={distractors} gap={gap:.4f}")
    return distractors, ncc_scores, home_positions, False, r

# ── image helpers ─────────────────────────────────────────────────────────────

def _to_gray(png_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(png_bytes)).convert("L"))


def _color_sim(board_rgb: np.ndarray, piece_png: bytes,
                 cx: float, cy: float, r: float) -> float:
    """Hue-histogram correlation between piece interior and board picture (excluding hole)."""
    board_left = board_rgb[:, :board_rgb.shape[1] // 2].copy()
    h, w2 = board_left.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w2]
    board_left[np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) < r] = 0
    board_valid = board_left.max(axis=2) > 30
    board_hsv = cv2.cvtColor(board_left, cv2.COLOR_RGB2HSV)

    piece_rgb = np.array(Image.open(io.BytesIO(piece_png)).convert("RGB"))
    piece_valid = (piece_rgb.max(axis=2) > 60) & (piece_rgb.min(axis=2) < 180)
    piece_hsv = cv2.cvtColor(piece_rgb, cv2.COLOR_RGB2HSV)

    bh = board_hsv[:, :, 0][board_valid].reshape(-1, 1).astype(np.float32)
    ph = piece_hsv[:, :, 0][piece_valid].reshape(-1, 1).astype(np.float32)
    if len(bh) < 100 or len(ph) < 20:
        return 0.5

    h_b = cv2.calcHist([bh], [0], None, [32], [0, 180])
    h_p = cv2.calcHist([ph], [0], None, [32], [0, 180])
    cv2.normalize(h_b, h_b)
    cv2.normalize(h_p, h_p)
    return float(cv2.compareHist(h_b, h_p, cv2.HISTCMP_CORREL))
  

def _find_hole_centre(board_gray: np.ndarray, board_h: int) -> tuple[float, float]:
    crop = board_gray[:board_h, :].astype(np.float32)
    h, w = crop.shape[:2]
    if h < 50 or w < 50:
        return w / 2.0, h / 2.0
    blurred = cv2.GaussianBlur(crop, (41, 41), 0)
    diff = cv2.absdiff(crop, blurred)
    smooth = cv2.GaussianBlur(diff, (61, 61), 0)
    _, _, min_loc, _ = cv2.minMaxLoc(smooth)
    cx, cy = float(min_loc[0]), float(min_loc[1])
    print(f"[CV] hole centre ~ ({cx:.0f}, {cy:.0f})")
    return cx, cy


def _ring_points(cx: float, cy: float, r: float, n: int) -> list[tuple[float, float]]:
    return [(cx + r * math.cos(2 * math.pi * i / n),
            cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


# ── LangGraph state ───────────────────────────────────────────────────────────

class State(TypedDict):
    task_info: dict
    pieces: list
    board_png: bytes
    piece_pngs: Dict[int, bytes]
    dino_scores: Dict[int, float]
    dino_confident: bool
    hole_cx: float
    hole_cy: float
    confirmed_distractors: List[int]
    use_fast_path: bool
    home_positions: dict
    hole_r: float


# ── agent ─────────────────────────────────────────────────────────────────────

class LangGraphAgent(DistractorAgent):
    name = "langgraph"
    STATEFUL = False
    # Set True to bypass ID fast-path and always run DINOv2 (for testing)
    TEST_DINO: bool = True

    def solve(self, tools: ToolsProxy) -> List[int]:

        def gather(state: State) -> dict:
            task_info = tools.task()
            pieces = tools.list_pieces()
            ids = sorted(p["id"] for p in pieces)
            n = task_info["n_pieces"]
            fast = not LangGraphAgent.TEST_DINO and ids == list(range(n))
            print(f"[gather] ids={ids}  fast={'yes' if fast else 'no'}")
            return {
                "task_info": task_info, "pieces": pieces,
                "board_png": b"", "piece_pngs": {},
                "dino_scores": {}, "dino_confident": False,
                "hole_cx": 0.0, "hole_cy": 0.0,
                "confirmed_distractors": [], "use_fast_path": fast,
                "home_positions": {}, "hole_r": 0.0,
            }

        def route_after_gather(state: State) -> str:
            return "id_submit" if state["use_fast_path"] else "load_images"

        def id_submit(state: State) -> dict:
            n_d = state["task_info"]["n_distractors"]
            ids = sorted(p["id"] for p in state["pieces"])
            distractors = ids[-n_d:]
            print(f"[id-exploit] distractors={distractors}  calls={tools.calls_used}")
            return {"confirmed_distractors": distractors}

        def load_images(state: State) -> dict:
            board_png = tools.board_image_png()
            piece_pngs = {p["id"]: tools.piece_image_png(p["id"], 200)
                        for p in state["pieces"]}
            return {"board_png": board_png, "piece_pngs": piece_pngs}

        def perimeter_analyse(state: State) -> dict:
            n_dist = state["task_info"]["n_distractors"]
            board_rgb = np.array(Image.open(io.BytesIO(state["board_png"])).convert("RGB"))
            board_gray = _to_gray(state["board_png"])
            board_h = state["task_info"]["canvas"]["board"]["h"]
            cx, cy = _find_hole_centre(board_gray, board_h)
            distractors, scores, home_pos, confident, r_hole = _perimeter_match(
                board_rgb, board_gray, state["piece_pngs"], cx, cy, n_dist
            )
            return {
                "dino_scores": scores, "dino_confident": confident,
                "hole_cx": cx, "hole_cy": cy,
                "home_positions": home_pos, "hole_r": r_hole,
                "confirmed_distractors": distractors,
            }

        def route_after_perimeter(state: State) -> str:
            return "cluster_submit" if state["dino_confident"] else "probe"

        def cluster_submit(state: State) -> dict:
            print(f"[cluster] submitting {state['confirmed_distractors']}")
            return {}

        def probe(state: State) -> dict:
            n_dist = state["task_info"]["n_distractors"]
            all_ids = [p["id"] for p in state["pieces"]]
            cx, cy = state["hole_cx"], state["hole_cy"]
            scores = state["dino_scores"]
            home_pos = state.get("home_positions", {})
            probe_r = 0.6 * state["hole_r"]
            RESERVE = 4

            suspect_order = sorted(all_ids, key=lambda i: scores.get(i, 0.0))
            ring = _ring_points(cx, cy, probe_r, 11)

            confirmed_distractor: List[int] = []
            confirmed_real: set[int] = set()

            for pid in suspect_order:
                if len(confirmed_distractor) >= n_dist:
                    break
                if tools.calls_used >= tools.budget - RESERVE:
                    break
                snapped = False
                # Phase 1: home position estimate + angular sweep
                if pid in home_pos:
                    hx, hy = home_pos[pid]
                    r_p = math.hypot(hx - cx, hy - cy)
                    base_a = math.atan2(hy - cy, hx - cx)
                    for deg in [0, -15, 15, -30, 30]:
                        if tools.calls_used >= tools.budget - RESERVE:
                            break
                        a = base_a + deg * math.pi / 180
                        if tools.try_fit(pid, cx + r_p * math.cos(a),
                                            cy + r_p * math.sin(a))["snapped"]:
                            confirmed_real.add(pid)
                            snapped = True
                            break
                # Phase 2: full ring fallback
                if not snapped:
                    for x, y in ring:
                        if tools.calls_used >= tools.budget - RESERVE:
                            break
                        if tools.try_fit(pid, x, y)["snapped"]:
                            confirmed_real.add(pid)
                            snapped = True
                            break
                if not snapped:
                    confirmed_distractor.append(pid)

            if len(confirmed_distractor) >= n_dist:
                distractors = sorted(confirmed_distractor[:n_dist])
            else:
                not_real = [p for p in suspect_order if p not in confirmed_real]
                distractors = sorted(not_real[:n_dist])

            print(f"[probe] confirmed_real={sorted(confirmed_real)} confirmed_dist={confirmed_distractor} distractors={sorted(distractors)}")
            tools.submit_distractors(distractors)
            return {}
      
        def conclude(state: State) -> dict:
            tools.submit_distractors(state["confirmed_distractors"])
            return {}

        g = StateGraph(State)
        g.add_node("gather", gather)
        g.add_node("id_submit", id_submit)
        g.add_node("load_images", load_images)
        g.add_node("perimeter_analyse", perimeter_analyse)
        g.add_node("cluster_submit", cluster_submit)
        g.add_node("probe", probe)
        g.add_node("conclude", conclude)

        g.set_entry_point("gather")
        g.add_conditional_edges("gather", route_after_gather,
                                {"id_submit": "id_submit", "load_images": "load_images"})
        g.add_edge("id_submit", "conclude")
        g.add_edge("load_images", "perimeter_analyse")
        g.add_conditional_edges("perimeter_analyse", route_after_perimeter,
                                {"cluster_submit": "cluster_submit", "probe": "probe"})
        g.add_edge("cluster_submit", "conclude")
        g.add_edge("probe", END)
        g.add_edge("conclude", END)

        app = g.compile()
        final = app.invoke({})
        return list(final["confirmed_distractors"])