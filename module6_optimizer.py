"""
Module 6: Multi-Period MILP Squad Optimizer & Execution Layer
==============================================================
Rolling N-gameweek squad selection with PuLP (CBC solver; swap to HiGHS via
solver=pulp.HiGHS_CMD() if installed for a large speed-up).

Implements:
  - Objective: discounted sum of starting-XI xPts + captaincy - 4*hits
  - Budget with 50% profit retention selling prices (Module 4 rule)
  - Formation (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD, XI of 11), squad 2-5-5-3
  - Max 3 per club
  - Free-transfer bank (up to 5), hits beyond FTs
  - Forced exclusions (Module 3 emergency trigger hook)
  - Bench ordering by conditional xPts + vice-captain by distinct kickoff
  - EO tilt hook (Module 7): per-player per-GW bonus/penalty vector

Python >= 3.10
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pulp

POSITIONS = ("GKP", "DEF", "MID", "FWD")
SQUAD_QUOTA = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
HIT_COST = 4
GAMMA = 0.9


@dataclass
class OptPlayer:
    id: int
    name: str
    team: str
    position: str
    buy_price: float                       # current market price (to buy)
    sell_price: Optional[float] = None     # 50%-retention price if owned; else None
    xpts: dict[int, float] = field(default_factory=dict)   # {gw: xPts}
    kickoff_slot: dict[int, int] = field(default_factory=dict)  # {gw: slot id} for VC logic
    eo_tilt: dict[int, float] = field(default_factory=dict)     # Module 7 adjustment (+/- pts)


@dataclass
class SolveResult:
    status: str
    objective: float
    squads: dict[int, list[int]]           # {gw: 15 player ids}
    starting: dict[int, list[int]]
    captain: dict[int, int]
    vice: dict[int, int]
    bench_order: dict[int, list[int]]
    transfers_in: dict[int, list[int]]
    transfers_out: dict[int, list[int]]
    hits: dict[int, int]


class SquadOptimizer:
    def __init__(self, players: list[OptPlayer], gameweeks: list[int],
                 current_squad: set[int], bank: float,
                 free_transfers: int = 1, gamma: float = GAMMA,
                 forced_out: Optional[set[int]] = None,
                 use_eo_tilt: bool = False):
        self.players = {p.id: p for p in players}
        self.gws = sorted(gameweeks)
        self.current_squad = set(current_squad)
        self.bank = bank
        self.ft0 = min(free_transfers, 5)
        self.gamma = gamma
        self.forced_out = forced_out or set()
        self.use_eo_tilt = use_eo_tilt

    def _adj_xpts(self, p: OptPlayer, gw: int) -> float:
        base = p.xpts.get(gw, 0.0)
        if self.use_eo_tilt:
            base += p.eo_tilt.get(gw, 0.0)
        return base

    def solve(self, solver: Optional[pulp.LpSolver] = None,
              time_limit: int = 60) -> SolveResult:
        P, G = list(self.players), self.gws
        m = pulp.LpProblem("fpl_multi_period", pulp.LpMaximize)

        sq = pulp.LpVariable.dicts("squad", (P, G), cat="Binary")     # in 15
        xi = pulp.LpVariable.dicts("start", (P, G), cat="Binary")     # in XI
        cp = pulp.LpVariable.dicts("cap", (P, G), cat="Binary")
        tin = pulp.LpVariable.dicts("tin", (P, G), cat="Binary")
        tout = pulp.LpVariable.dicts("tout", (P, G), cat="Binary")
        ft = pulp.LpVariable.dicts("ft", G, lowBound=0, upBound=5, cat="Integer")
        hits = pulp.LpVariable.dicts("hits", G, lowBound=0, cat="Integer")

        # ---------------- objective ----------------
        m += pulp.lpSum(
            (self.gamma ** t) * (
                pulp.lpSum(self._adj_xpts(self.players[i], g) * xi[i][g] for i in P)
                + pulp.lpSum(self._adj_xpts(self.players[i], g) * cp[i][g] for i in P)
                - HIT_COST * hits[g]
            )
            for t, g in enumerate(G)
        )

        # ---------------- constraints ----------------
        for g in G:
            m += pulp.lpSum(sq[i][g] for i in P) == 15
            m += pulp.lpSum(xi[i][g] for i in P) == 11
            m += pulp.lpSum(cp[i][g] for i in P) == 1
            for pos in POSITIONS:
                ids = [i for i in P if self.players[i].position == pos]
                m += pulp.lpSum(sq[i][g] for i in ids) == SQUAD_QUOTA[pos]
                m += pulp.lpSum(xi[i][g] for i in ids) >= XI_MIN[pos]
                m += pulp.lpSum(xi[i][g] for i in ids) <= XI_MAX[pos]
            clubs = {self.players[i].team for i in P}
            for c in clubs:
                m += pulp.lpSum(sq[i][g] for i in P
                                if self.players[i].team == c) <= MAX_PER_CLUB
            for i in P:
                m += xi[i][g] <= sq[i][g]
                m += cp[i][g] <= xi[i][g]

        # transfer flow: squad evolves from current squad via tin/tout
        for i in P:
            g0 = G[0]
            was_in = 1 if i in self.current_squad else 0
            m += sq[i][g0] == was_in + tin[i][g0] - tout[i][g0]
            for k in range(1, len(G)):
                m += sq[i][G[k]] == sq[i][G[k - 1]] + tin[i][G[k]] - tout[i][G[k]]
            if i in self.forced_out:            # Module 3 emergency exclusion
                for g in G:
                    m += sq[i][g] == 0

        # budget per GW: bank rolls; sell at retention price, buy at market
        bank_prev = self.bank
        bank_vars = {}
        for g in G:
            bank_vars[g] = pulp.LpVariable(f"bank_{g}", lowBound=0)
            sell_val = pulp.lpSum(
                (self.players[i].sell_price or self.players[i].buy_price) * tout[i][g]
                for i in P)
            buy_val = pulp.lpSum(self.players[i].buy_price * tin[i][g] for i in P)
            prev = bank_prev if g == G[0] else bank_vars[G[G.index(g) - 1]]
            m += bank_vars[g] == prev + sell_val - buy_val

        # free-transfer bank + hits: ft_{g+1} = min(5, ft_g - used + 1) linearized
        m += ft[G[0]] == self.ft0
        # Initial draft (pre-season / wildcard): building the first 15 is free.
        initial_build = not self.current_squad
        for k, g in enumerate(G):
            n_tr = pulp.lpSum(tin[i][g] for i in P)
            if k == 0 and initial_build:
                m += hits[g] == 0        # the first 15 cost nothing...
                used = 0                 # ...and consume no free transfers
            else:
                m += hits[g] >= n_tr - ft[g]
                m += hits[g] <= n_tr                  # can't buy FTs with hits
                used = n_tr - hits[g]
            if k < len(G) - 1:
                nxt = G[k + 1]
                m += ft[nxt] <= ft[g] - used + 1
                m += ft[nxt] <= 5
                m += ft[nxt] >= 1

        solver = solver or pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
        m.solve(solver)

        return self._extract(m, sq, xi, cp, tin, tout, hits)

    # ------------------------------------------------------------------
    def _extract(self, m, sq, xi, cp, tin, tout, hits) -> SolveResult:
        P, G = list(self.players), self.gws
        v = lambda var: var.value() is not None and var.value() > 0.5
        squads = {g: [i for i in P if v(sq[i][g])] for g in G}
        starting = {g: [i for i in P if v(xi[i][g])] for g in G}
        captain = {g: next((i for i in P if v(cp[i][g])), -1) for g in G}

        bench_order, vice = {}, {}
        for g in G:
            bench = [i for i in squads[g] if i not in starting[g]]
            bench_out = [i for i in bench if self.players[i].position != "GKP"]
            bench_gk = [i for i in bench if self.players[i].position == "GKP"]
            bench_out.sort(key=lambda i: self._adj_xpts(self.players[i], g), reverse=True)
            bench_order[g] = bench_gk + bench_out

            # VC: highest xPts starter whose kickoff slot differs from captain's
            cap = captain[g]
            cap_slot = self.players[cap].kickoff_slot.get(g) if cap in self.players else None
            cands = sorted((i for i in starting[g] if i != cap),
                           key=lambda i: self._adj_xpts(self.players[i], g), reverse=True)
            distinct = [i for i in cands
                        if cap_slot is None
                        or self.players[i].kickoff_slot.get(g) != cap_slot]
            vice[g] = (distinct or cands)[0] if cands else -1

        return SolveResult(
            status=pulp.LpStatus[m.status],
            objective=round(pulp.value(m.objective) or 0.0, 2),
            squads=squads, starting=starting, captain=captain, vice=vice,
            bench_order=bench_order,
            transfers_in={g: [i for i in P if v(tin[i][g])] for g in G},
            transfers_out={g: [i for i in P if v(tout[i][g])] for g in G},
            hits={g: int(hits[g].value() or 0) for g in G},
        )


# ---------------------------------------------------------------------------
# Example execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    random.seed(7)
    teams = ["ARS", "LIV", "MCI", "CHE", "TOT", "NEW", "AVL", "BHA", "WHU", "EVE"]
    gws = [1, 2, 3]
    pool: list[OptPlayer] = []
    pid = 1
    for pos, n, lo, hi, base in (("GKP", 8, 4.0, 5.5, 3.2), ("DEF", 20, 4.0, 6.5, 3.6),
                                 ("MID", 24, 4.5, 13.0, 4.2), ("FWD", 12, 4.5, 12.5, 4.0)):
        for _ in range(n):
            price = round(random.uniform(lo, hi), 1)
            quality = (price - lo) / (hi - lo)
            pool.append(OptPlayer(
                id=pid, name=f"{pos}{pid}", team=random.choice(teams), position=pos,
                buy_price=price,
                xpts={g: round(max(0.5, random.gauss(base + 3.5 * quality, 0.8)), 2)
                      for g in gws},
                kickoff_slot={g: random.randint(1, 4) for g in gws},
            ))
            pid += 1

    # Start from the 15 cheapest legal players; give ourselves 3.0 in the bank
    by_pos = {p: sorted((x for x in pool if x.position == p), key=lambda x: x.buy_price)
              for p in POSITIONS}
    current = ([x.id for x in by_pos["GKP"][:2]] + [x.id for x in by_pos["DEF"][:5]]
               + [x.id for x in by_pos["MID"][:5]] + [x.id for x in by_pos["FWD"][:3]])
    for p in pool:
        if p.id in current:
            p.sell_price = p.buy_price       # no profit yet

    opt = SquadOptimizer(pool, gws, current_squad=set(current), bank=3.0,
                         free_transfers=1)
    res = opt.solve(time_limit=30)

    name = lambda i: opt.players[i].name
    print(f"Status: {res.status}   Objective: {res.objective}\n")
    for g in gws:
        print(f"GW{g}: in={list(map(name, res.transfers_in[g]))} "
              f"out={list(map(name, res.transfers_out[g]))} hits={res.hits[g]}")
        print(f"     C={name(res.captain[g])}  VC={name(res.vice[g])}  "
              f"bench={list(map(name, res.bench_order[g]))}")
