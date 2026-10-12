"""Graph / mule-ring layer.

The insight: a payee that shares a device fingerprint, phone, or IP with
previously-flagged accounts is suspicious even when every per-transaction
feature looks clean. Per-transaction detectors cannot see this — one clean
transaction from an account that is 1 of 40 in a ring looks exactly like a
genuine grocery payment.

Implemented with adjacency sets + BFS (stdlib only). NetworkX would be nicer to
draw but the traversal is trivial, and keeping the dependency list short matters
more for a repo someone has to run.

The `seed_flagged` design note: in production the flagged-seed set would come
from a persisted blocklist. Here it comes from the rule/label layer so the demo
can show the "one bad seed lights up 39 clean accounts" effect without
requiring the model to already be trained.
"""

from __future__ import annotations

from collections import defaultdict, deque


class RingDetector:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.shared_device_weight = cfg.get("shared_device_weight", 0.55)
        self.shared_phone_weight = cfg.get("shared_phone_weight", 0.35)
        self.shared_ip_weight = cfg.get("shared_ip_weight", 0.25)
        self.min_size = cfg.get("ring_min_size", 3)
        self.max_component = cfg.get("max_component", 5000)

        # identifier -> set of account VPAs seen with it
        self.by_device: dict[str, set[str]] = defaultdict(set)
        self.by_phone: dict[str, set[str]] = defaultdict(set)
        self.by_ip: dict[str, set[str]] = defaultdict(set)
        # account -> its direct (identifier, kind) links. Without this reverse
        # index, finding an account's neighbours means scanning every identifier
        # table, which turns a per-transaction lookup into O(dataset) and blows
        # up as soon as adversarial rows share identifiers with many accounts.
        self.links: dict[str, set[tuple[str, str]]] = defaultdict(set)
        self.flagged: set[str] = set()
        self.edges: list[tuple[str, str, str]] = []  # (account, identifier, kind)

    def ingest(self, txns: list[dict], seed_accounts: set[str] | None = None) -> "RingDetector":
        """Index every account, and mark a seed-blocklist of known-bad accounts.

        `seed_accounts` is what makes this honest. Reading the ground-truth
        `label` field here would let the graph layer "discover" fraud using the
        answer key, and the reported numbers would be meaningless. In deployment
        these seeds come from accounts the rule layer blocked earlier — see
        `engine.seed_from_blocks`, which does exactly that on a first pass.
        """
        for t in txns:
            acct = t.get("user_home_vpa") or t.get("payee_vpa")
            if not acct:
                continue
            for kind, field, table in (
                ("device", "user_device_id", self.by_device),
                ("phone", "user_phone", self.by_phone),
                ("ip", "user_ip", self.by_ip),
            ):
                ident = t.get(field)
                if ident:
                    table[ident].add(acct)
                    self.links[acct].add((ident, kind))
                    self.edges.append((acct, ident, kind))
        if seed_accounts:
            self.flagged |= seed_accounts
        return self

    def _neighbours(self, acct: str) -> set[str]:
        """Accounts one hop away, via the reverse index."""
        out: set[str] = set()
        for ident, _kind in self.links.get(acct, ()):
            out |= self.by_device.get(ident) or set()
            out |= self.by_phone.get(ident) or set()
            out |= self.by_ip.get(ident) or set()
        return out - {acct}

    def risk(self, txn: dict, flag_threshold: int = 1) -> tuple[float, dict]:
        """Graph risk for one transaction + the evidence behind it.

        Returns (score 0..1, detail). Score is max of:
          - shared-identifier overlap with the flagged set (weighted per kind)
          - size of the connected component if it contains a flagged account
        """
        acct = txn.get("user_home_vpa") or txn.get("payee_vpa")
        if not acct:
            return 0.0, {"flagged_neighbours": 0, "component_size": 0}

        flagged_neigh: set[str] = set()
        for ident, _kind in self.links.get(acct, ()):
            flagged_neigh |= (self.by_device.get(ident) or set()) & self.flagged
            flagged_neigh |= (self.by_phone.get(ident) or set()) & self.flagged
            flagged_neigh |= (self.by_ip.get(ident) or set()) & self.flagged
        flagged_neigh.discard(acct)

        # Component size via BFS, bounded. A shared identifier (say a corporate
        # NAT IP) can fuse thousands of unrelated accounts into one component;
        # walking it fully on a per-transaction path is quadratic and, worse,
        # makes component_size meaningless. Cap the walk and stop early once the
        # size is big enough to saturate the score.
        cap = self.max_component
        seen = {acct}
        q = deque([acct])
        comp = 1
        hit_flagged = bool(flagged_neigh)
        while q and comp < cap:
            cur = q.popleft()
            for nb in self._neighbours(cur):
                if nb not in seen:
                    seen.add(nb)
                    comp += 1
                    if nb in self.flagged:
                        hit_flagged = True
                    q.append(nb)
            if hit_flagged and comp >= self.min_size:
                # enough evidence already; size beyond the cap adds nothing
                break

        seed_score = 0.0
        detail_flags: list[str] = sorted(flagged_neigh)
        if flagged_neigh:
            seed_score = self.shared_device_weight  # device overlap dominates
            # more flagged neighbours = stronger, capped at 1
            seed_score = min(1.0, seed_score + 0.1 * (len(flagged_neigh) - 1))

        comp_score = 0.0
        if comp >= self.min_size and hit_flagged:
            comp_score = min(0.9, 0.3 + 0.1 * (comp - self.min_size))

        score = max(seed_score, comp_score)
        detail = {
            "flagged_neighbours": len(detail_flags),
            "flagged_accounts": detail_flags[:5],
            "component_size": comp,
        }
        return score, detail

    def top_rings(self, top_k: int = 5) -> list[dict]:
        """Largest connected components, for the dashboard graph view."""
        all_accts = set()
        for table in (self.by_device, self.by_phone, self.by_ip):
            for accounts in table.values():
                all_accts |= accounts
        visited: set[str] = set()
        comps: list[list[str]] = []
        for a in all_accts:
            if a in visited:
                continue
            seen = {a}
            q = deque([a])
            while q and len(seen) < self.max_component:
                cur = q.popleft()
                for nb in self._neighbours(cur):
                    if nb not in seen:
                        seen.add(nb)
                        q.append(nb)
            visited |= seen
            comps.append(sorted(seen))
        comps.sort(key=len, reverse=True)
        out = []
        for c in comps[:top_k]:
            flagged_n = sum(1 for a in c if a in self.flagged)
            out.append({"size": len(c), "flagged": flagged_n, "accounts": c[:8]})
        return [c for c in out if c["size"] >= self.min_size]
