"""
AI Knockout competitive chess agent.

Submission entrypoint:
    choose_move(board_fen: str, legal_moves: list[str], time_left_ms: int) -> str

The engine is intentionally dependency-free: it implements a compact chess
position/search core and uses the competition-provided legal move list as the
final root legality authority.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

INF = 10**9
MATE = 10**8

# Piece values in centipawns.
VALUE = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0}

# Piece-square tables, indexed from White's perspective (a1 = 0).
PST = {
    "P": [
        0,0,0,0,0,0,0,0,
        5,10,10,-20,-20,10,10,5,
        5,-5,-10,0,0,-10,-5,5,
        0,0,0,20,20,0,0,0,
        5,5,10,25,25,10,5,5,
        10,10,20,30,30,20,10,10,
        50,50,50,50,50,50,50,50,
        0,0,0,0,0,0,0,0,
    ],
    "N": [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,0,5,5,0,-20,-40,
        -30,5,10,15,15,10,5,-30,
        -30,0,15,20,20,15,0,-30,
        -30,5,15,20,20,15,5,-30,
        -30,0,10,15,15,10,0,-30,
        -40,-20,0,0,0,0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    "B": [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,5,0,0,0,0,5,-10,
        -10,10,10,10,10,10,10,-10,
        -10,0,10,10,10,10,0,-10,
        -10,5,5,10,10,5,5,-10,
        -10,0,5,10,10,5,0,-10,
        -10,0,0,0,0,0,0,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    "R": [
        0,0,0,5,5,0,0,0,
        -5,0,0,0,0,0,0,-5,
        -5,0,0,0,0,0,0,-5,
        -5,0,0,0,0,0,0,-5,
        -5,0,0,0,0,0,0,-5,
        -5,0,0,0,0,0,0,-5,
        5,10,10,10,10,10,10,5,
        0,0,0,0,0,0,0,0,
    ],
    "Q": [
        -20,-10,-10,0,0,-10,-10,-20,
        -10,0,0,0,0,0,0,-10,
        -10,0,5,5,5,5,0,-10,
        0,0,5,5,5,5,0,0,
        -5,0,5,5,5,5,0,-5,
        -10,5,5,5,5,5,0,-10,
        -10,0,5,0,0,0,0,-10,
        -20,-10,-10,0,0,-10,-10,-20,
    ],
    "K": [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        20,20,0,0,0,0,20,20,
        20,30,10,0,0,10,30,20,
    ],
}

KNIGHT_DIRS = ((1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2))
KING_DIRS = ((1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1))
BISHOP_DIRS = ((1,1),(1,-1),(-1,1),(-1,-1))
ROOK_DIRS = ((1,0),(-1,0),(0,1),(0,-1))
QUEEN_DIRS = BISHOP_DIRS + ROOK_DIRS


@dataclass(slots=True)
class Move:
    frm: int
    to: int
    promotion: str = ""
    castle: bool = False
    ep: bool = False

    def uci(self) -> str:
        s = sq_name(self.frm) + sq_name(self.to)
        return s + (self.promotion.lower() if self.promotion else "")


@dataclass(slots=True)
class Undo:
    captured: str
    castling: str
    ep: int
    halfmove: int
    fullmove: int
    moved: str


def sq_name(s: int) -> str:
    return chr(97 + (s & 7)) + str((s >> 4) + 1)


def parse_sq(s: str) -> int:
    return (ord(s[0]) - 97) + ((ord(s[1]) - 49) << 4)


def color_of(p: str) -> int:
    return 1 if p.isupper() else -1


def opponent(c: int) -> int:
    return -c


class Board:
    def __init__(self, fen: str):
        parts = fen.split()
        board_part = parts[0]
        self.side = 1 if len(parts) < 2 or parts[1] == "w" else -1
        self.castling = parts[2] if len(parts) > 2 else "-"
        self.ep = -1 if len(parts) < 4 or parts[3] == "-" else parse_sq(parts[3])
        self.halfmove = int(parts[4]) if len(parts) > 4 else 0
        self.fullmove = int(parts[5]) if len(parts) > 5 else 1
        self.s = ["."] * 128
        rank = 7
        file = 0
        for ch in board_part:
            if ch == "/":
                rank -= 1
                file = 0
            elif ch.isdigit():
                file += int(ch)
            else:
                self.s[rank * 16 + file] = ch
                file += 1

    def clone(self) -> "Board":
        b = object.__new__(Board)
        b.s = self.s.copy()
        b.side, b.castling, b.ep = self.side, self.castling, self.ep
        b.halfmove, b.fullmove = self.halfmove, self.fullmove
        return b

    def king_square(self, side: int) -> int:
        k = "K" if side == 1 else "k"
        for i in range(128):
            if (i & 15) == 8:
                continue
            if self.s[i] == k:
                return i
        return -1

    def attacked(self, sq: int, by: int) -> bool:
        # Pawns
        r, f = sq >> 4, sq & 7
        if by == 1:
            for df in (-1, 1):
                x = sq - 16 + df
                if 0 <= x < 128 and (x & 8) == 0 and self.s[x] == "P":
                    return True
        else:
            for df in (-1, 1):
                x = sq + 16 + df
                if 0 <= x < 128 and (x & 8) == 0 and self.s[x] == "p":
                    return True

        # Knights
        nk = "N" if by == 1 else "n"
        for df, dr in KNIGHT_DIRS:
            x = (f + df) + ((r + dr) << 4)
            if 0 <= x < 128 and (x & 8) == 0 and self.s[x] == nk:
                return True

        # King
        kk = "K" if by == 1 else "k"
        for df, dr in KING_DIRS:
            x = (f + df) + ((r + dr) << 4)
            if 0 <= x < 128 and (x & 8) == 0 and self.s[x] == kk:
                return True

        # Sliding pieces
        for dirs, chars in (
            (BISHOP_DIRS, ("B", "Q") if by == 1 else ("b", "q")),
            (ROOK_DIRS, ("R", "Q") if by == 1 else ("r", "q")),
        ):
            for df, dr in dirs:
                x = f + df
                rr = r + dr
                while 0 <= x < 8 and 0 <= rr < 8:
                    z = x + rr * 16
                    p = self.s[z]
                    if p != ".":
                        if p in chars:
                            return True
                        break
                    x += df
                    rr += dr
        return False

    def in_check(self, side: Optional[int] = None) -> bool:
        side = self.side if side is None else side
        k = self.king_square(side)
        return k < 0 or self.attacked(k, -side)

    def pseudo_moves(self, captures_only: bool = False):
        side = self.side
        for i, p in enumerate(self.s):
            if (i & 8) or p == "." or color_of(p) != side:
                continue
            pt = p.upper()
            r, f = i >> 4, i & 7

            if pt == "P":
                dr = 1 if side == 1 else -1
                start = 1 if side == 1 else 6
                promo_rank = 7 if side == 1 else 0

                one = i + dr * 16
                if not captures_only and 0 <= one < 128 and (one & 8) == 0 and self.s[one] == ".":
                    if (one >> 4) == promo_rank:
                        for pr in "qrbn":
                            yield Move(i, one, pr)
                    else:
                        yield Move(i, one)
                        two = i + dr * 32
                        if r == start and self.s[two] == ".":
                            yield Move(i, two)
                for df in (-1, 1):
                    nf = f + df
                    nr = r + dr
                    if not (0 <= nf < 8 and 0 <= nr < 8):
                        continue
                    to = nf + nr * 16
                    target = self.s[to]
                    if target != "." and color_of(target) == -side:
                        if nr == promo_rank:
                            for pr in "qrbn":
                                yield Move(i, to, pr)
                        else:
                            yield Move(i, to)
                    elif to == self.ep:
                        yield Move(i, to, ep=True)

            elif pt == "N":
                for df, dr in KNIGHT_DIRS:
                    nf, nr = f + df, r + dr
                    if 0 <= nf < 8 and 0 <= nr < 8:
                        to = nf + nr * 16
                        if self.s[to] == ".":
                            if not captures_only:
                                yield Move(i, to)
                        elif color_of(self.s[to]) == -side:
                            yield Move(i, to)

            elif pt in ("B", "R", "Q"):
                dirs = BISHOP_DIRS if pt == "B" else ROOK_DIRS if pt == "R" else QUEEN_DIRS
                for df, dr in dirs:
                    nf, nr = f + df, r + dr
                    while 0 <= nf < 8 and 0 <= nr < 8:
                        to = nf + nr * 16
                        target = self.s[to]
                        if target == ".":
                            if not captures_only:
                                yield Move(i, to)
                        else:
                            if color_of(target) == -side:
                                yield Move(i, to)
                            break
                        nf += df
                        nr += dr

            elif pt == "K":
                for df, dr in KING_DIRS:
                    nf, nr = f + df, r + dr
                    if 0 <= nf < 8 and 0 <= nr < 8:
                        to = nf + nr * 16
                        if self.s[to] == ".":
                            if not captures_only:
                                yield Move(i, to)
                        elif color_of(self.s[to]) == -side:
                            yield Move(i, to)
                if not captures_only and not self.in_check(side):
                    if side == 1 and i == parse_sq("e1"):
                        if "K" in self.castling and self.s[parse_sq("f1")] == "." and self.s[parse_sq("g1")] == ".":
                            if not self.attacked(parse_sq("f1"), -side) and not self.attacked(parse_sq("g1"), -side):
                                yield Move(i, parse_sq("g1"), castle=True)
                        if "Q" in self.castling and self.s[parse_sq("d1")] == "." and self.s[parse_sq("c1")] == "." and self.s[parse_sq("b1")] == ".":
                            if not self.attacked(parse_sq("d1"), -side) and not self.attacked(parse_sq("c1"), -side):
                                yield Move(i, parse_sq("c1"), castle=True)
                    elif side == -1 and i == parse_sq("e8"):
                        if "k" in self.castling and self.s[parse_sq("f8")] == "." and self.s[parse_sq("g8")] == ".":
                            if not self.attacked(parse_sq("f8"), -side) and not self.attacked(parse_sq("g8"), -side):
                                yield Move(i, parse_sq("g8"), castle=True)
                        if "q" in self.castling and self.s[parse_sq("d8")] == "." and self.s[parse_sq("c8")] == "." and self.s[parse_sq("b8")] == ".":
                            if not self.attacked(parse_sq("d8"), -side) and not self.attacked(parse_sq("c8"), -side):
                                yield Move(i, parse_sq("c8"), castle=True)

    def push(self, m: Move) -> Undo:
        moving = self.s[m.frm]
        captured = self.s[m.to]
        u = Undo(captured, self.castling, self.ep, self.halfmove, self.fullmove, moving)
        self.s[m.frm] = "."
        self.s[m.to] = moving

        if m.ep:
            cap_sq = m.to - 16 if self.side == 1 else m.to + 16
            u.captured = self.s[cap_sq]
            self.s[cap_sq] = "."

        if m.promotion:
            self.s[m.to] = m.promotion.upper() if self.side == 1 else m.promotion.lower()

        if m.castle:
            if m.to > m.frm:  # king side
                rf, rt = m.frm + 3, m.frm + 1
            else:
                rf, rt = m.frm - 4, m.frm - 1
            self.s[rt] = self.s[rf]
            self.s[rf] = "."

        # Castling rights
        rights = self.castling
        if moving.upper() == "K":
            rights = rights.replace("K", "").replace("Q", "") if self.side == 1 else rights.replace("k", "").replace("q", "")
        if m.frm == parse_sq("h1") or m.to == parse_sq("h1"):
            rights = rights.replace("K", "")
        if m.frm == parse_sq("a1") or m.to == parse_sq("a1"):
            rights = rights.replace("Q", "")
        if m.frm == parse_sq("h8") or m.to == parse_sq("h8"):
            rights = rights.replace("k", "")
        if m.frm == parse_sq("a8") or m.to == parse_sq("a8"):
            rights = rights.replace("q", "")
        self.castling = rights or "-"

        self.ep = -1
        if moving.upper() == "P" and abs(m.to - m.frm) == 32:
            self.ep = (m.to + m.frm) // 2

        self.halfmove = 0 if moving.upper() == "P" or captured != "." or m.ep else self.halfmove + 1
        if self.side == -1:
            self.fullmove += 1
        self.side = -self.side
        return u

    def pop(self, m: Move, u: Undo):
        self.side = -self.side
        if self.side == -1:
            self.fullmove -= 1
        self.halfmove, self.castling, self.ep = u.halfmove, u.castling, u.ep

        moving = u.moved
        self.s[m.frm] = moving
        self.s[m.to] = u.captured

        if m.ep:
            cap_sq = m.to - 16 if self.side == 1 else m.to + 16
            self.s[cap_sq] = "p" if self.side == 1 else "P"

        if m.castle:
            if m.to > m.frm:
                rf, rt = m.frm + 3, m.frm + 1
            else:
                rf, rt = m.frm - 4, m.frm - 1
            self.s[rf] = self.s[rt]
            self.s[rt] = "."

    def legal_moves(self, captures_only=False):
        for m in self.pseudo_moves(captures_only):
            u = self.push(m)
            illegal = self.in_check(-self.side)
            self.pop(m, u)
            if not illegal:
                yield m


# Search state.
_DEADLINE = 0.0
_NODES = 0
_TT: dict[tuple, tuple[int, int, int, Optional[str]]] = {}


def board_key(b: Board):
    return (tuple(b.s), b.side, b.castling, b.ep)


def evaluate(b: Board) -> int:
    score = 0
    white_material = black_material = 0
    pieces = 0

    for sq, p in enumerate(b.s):
        if (sq & 8) or p == ".":
            continue
        pieces += 1
        v = VALUE[p.upper()]
        if p.isupper():
            white_material += v
            idx = sq
            score += v + PST[p.upper()][idx]
        else:
            idx = (7 - (sq >> 4)) * 16 + (sq & 7)
            # Mirror ranks for Black.
            idx = (7 - (sq >> 4)) * 8 + (sq & 7)
            score -= v + PST[p.upper()][idx]

    # Mobility and center control are useful tie-breakers.
    old = b.side
    wm = bm = 0
    center = 0
    for sq in (parse_sq("d4"), parse_sq("e4"), parse_sq("d5"), parse_sq("e5")):
        p = b.s[sq]
        if p == ".":
            continue
        center += 12 if p.isupper() else -12

    # Approximate king safety: penalize exposed kings, reward nearby attackers.
    wk = b.king_square(1)
    bk = b.king_square(-1)
    if wk >= 0:
        wr, wf = wk >> 4, wk & 7
        if wr <= 1:
            score += 18
        elif wr >= 5:
            score -= 12
    if bk >= 0:
        br, bf = bk >> 4, bk & 7
        if br >= 6:
            score -= 18
        elif br <= 2:
            score += 12

    # Castling rights have modest value.
    if "K" in b.castling or "Q" in b.castling:
        score += 12
    if "k" in b.castling or "q" in b.castling:
        score -= 12

    score += center

    # Side-to-move mobility.
    try:
        mob = sum(1 for _ in b.legal_moves())
        score += 3 * mob if b.side == 1 else -3 * mob
    except Exception:
        pass

    return score


def move_capture_value(b: Board, m: Move) -> int:
    p = b.s[m.frm]
    victim = b.s[m.to]
    if m.ep:
        victim = "p" if b.side == 1 else "P"
    return VALUE.get(victim.upper(), 0) - VALUE.get(p.upper(), 0) // 10


def gives_check(b: Board, m: Move) -> bool:
    u = b.push(m)
    result = b.in_check(b.side)
    b.pop(m, u)
    return result


def ordered_moves(b: Board, moves, tt_move: Optional[str] = None):
    scored = []
    for m in moves:
        uci = m.uci()
        score = 0
        if uci == tt_move:
            score += 10_000_000
        if b.s[m.to] != "." or m.ep:
            score += 1_000_000 + move_capture_value(b, m) * 100
        if m.promotion:
            score += 900_000 + VALUE[m.promotion.upper()] * 100
        try:
            if gives_check(b, m):
                score += 500_000
        except Exception:
            pass
        # Develop toward the center / castle.
        if b.s[m.frm].upper() in ("N", "B") and (m.to & 7) in (2,3,4,5):
            score += 4_000
        if m.castle:
            score += 8_000
        scored.append((score, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored]


def quiescence(b: Board, alpha: int, beta: int) -> int:
    global _NODES
    _NODES += 1
    if time.perf_counter() >= _DEADLINE:
        raise TimeoutError

    stand = evaluate(b)
    if b.side == -1:
        stand = -stand

    if stand >= beta:
        return beta
    if stand > alpha:
        alpha = stand

    moves = list(b.legal_moves(captures_only=True))
    for m in ordered_moves(b, moves):
        u = b.push(m)
        score = -quiescence(b, -beta, -alpha)
        b.pop(m, u)
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def negamax(b: Board, depth: int, alpha: int, beta: int, ply: int = 0) -> int:
    global _NODES
    _NODES += 1
    if time.perf_counter() >= _DEADLINE:
        raise TimeoutError

    key = board_key(b)
    entry = _TT.get(key)
    tt_move = None
    if entry is not None:
        edepth, flag, val, tt_move = entry
        if edepth >= depth:
            if flag == 0:
                return val
            if flag == 1:
                alpha = max(alpha, val)
            elif flag == 2:
                beta = min(beta, val)
            if alpha >= beta:
                return val

    moves = list(b.legal_moves())
    if not moves:
        if b.in_check():
            return -MATE + ply
        return 0

    if depth <= 0:
        return quiescence(b, alpha, beta)

    original_alpha = alpha
    best = -INF
    best_move = None

    for m in ordered_moves(b, moves, tt_move):
        u = b.push(m)
        score = -negamax(b, depth - 1, -beta, -alpha, ply + 1)
        b.pop(m, u)

        if score > best:
            best, best_move = score, m
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    flag = 0 if best > original_alpha and best < beta else 1 if best <= original_alpha else 2
    _TT[key] = (depth, flag, best, best_move.uci() if best_move else None)
    return best


def _parse_uci(u: str) -> Move:
    return Move(parse_sq(u[:2]), parse_sq(u[2:4]), u[4] if len(u) > 4 else "")


def _root_search(b: Board, root_moves, depth: int):
    alpha, beta = -INF, INF
    best_score = -INF
    best = root_moves[0]

    # Root moves are supplied by the competition and are therefore authoritative.
    # We search them through our internal board for strength.
    for m in root_moves:
        if time.perf_counter() >= _DEADLINE:
            raise TimeoutError
        u = b.push(m)
        score = -negamax(b, depth - 1, -beta, -alpha, 1)
        b.pop(m, u)
        if score > best_score:
            best_score, best = score, m
        alpha = max(alpha, score)
    return best, best_score


def choose_move(board_fen: str, legal_moves: list[str], time_left_ms: int) -> str:
    """
    Choose and return a legal UCI move.

    The server remains the final authority for legality. The supplied legal_moves
    list is filtered against our search result before returning.
    """
    if not legal_moves:
        raise ValueError("No legal moves supplied")

    # Fail-safe: always have a server-approved move available.
    fallback = legal_moves[0]

    try:
        b = Board(board_fen)
    except Exception:
        return fallback

    allowed = set(legal_moves)
    root = []
    for u in legal_moves:
        try:
            m = _parse_uci(u)
            # We trust the server's root list, but avoid malformed input.
            if len(u) >= 4 and 0 <= m.frm < 128 and 0 <= m.to < 128:
                root.append(m)
        except Exception:
            pass
    if not root:
        return fallback

    # Tactical urgency gets more time. Never consume the entire clock.
    budget_ms = max(20, min(max(25, time_left_ms // 12), 1800))
    if time_left_ms < 1500:
        budget_ms = max(10, time_left_ms // 5)
    global _DEADLINE, _NODES, _TT
    _DEADLINE = time.perf_counter() + budget_ms / 1000.0
    _NODES = 0

    # Keep TT bounded so a long match does not grow without limit.
    if len(_TT) > 120_000:
        _TT.clear()

    # Immediate mate/check/capture scan before deeper search.
    try:
        for m in root:
            u = b.push(m)
            is_mate = b.in_check(b.side) and not any(b.legal_moves())
            gives = b.in_check(b.side)
            b.pop(m, u)
            if is_mate:
                return m.uci() if m.uci() in allowed else fallback
            if gives and len(root) > 1:
                # Keep checking candidates in search; this is only a priority hint.
                pass
    except Exception:
        pass

    best = root[0]
    # Iterative deepening. Depth 1 is nearly always completed, deeper searches
    # are retained only if they finish before the deadline.
    for depth in range(1, 32):
        try:
            candidate, _score = _root_search(b, root, depth)
            if candidate.uci() in allowed:
                best = candidate
        except TimeoutError:
            break
        except Exception:
            break

        # If we found a forced mate score, no deeper search is needed.
        # The mate score is encoded in the TT/root result but we intentionally
        # keep the stop condition conservative for reliability.
        if time.perf_counter() >= _DEADLINE:
            break

    uci = best.uci()
    return uci if uci in allowed else fallback
