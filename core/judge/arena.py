"""The arena / judge: runs the full attack-defense round-robin.

Pipeline per sample (spec section 四):
  1. load clean DB (gold docs + top-M distractors from the global corpus)
  2. AttackContext(question, attack_target, blackbox) -> attack.attack() -> poison
  3. db.add_poison(poison)
  4. DefenseEnv(question, DefenseDBProxy(db)) -> defense.defend()
  5. RAG retrieve Top-K -> PromptBuilder -> LLM -> response
  6. Evaluator: AttackSuccess / DefenseSuccess

CleanAccuracy is measured by running the same defense on an unpoisoned DB.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from tqdm import tqdm

from core.config import Config
from core.data_loader import load_corpus, load_samples
from core.database.knowledge_db import DefenseDBProxy, KnowledgeDB
from core.env.env import AttackContext, DefenseEnv
from core.judge.evaluator import Evaluator
from core.llm.llm_client import LLMClient
from core.rag.prompt_builder import PromptBuilder
from core.rag.rag_system import RAGSystem
from core.rag.retriever import build_retriever


class Arena:
    def __init__(self, config: Optional[Config] = None,
                 retriever_backend: Optional[str] = None):
        self.config = config or Config()
        self.corpus = load_corpus(self.config.dataset["corpus_path"])
        if not self.corpus:
            raise RuntimeError(
                f"Corpus not found at {self.config.dataset['corpus_path']}. "
                "Run `python scripts/build_local_corpus.py` first."
            )
        rcfg = dict(self.config.retriever)
        if retriever_backend:
            rcfg["backend"] = retriever_backend
        self.retriever = build_retriever(rcfg, corpus_texts=[d["text"] for d in self.corpus])
        self.llm = LLMClient(
            config_path=self.config.llm.get("config_path"),
            allow_mock_fallback=self.config.llm.get("allow_mock_fallback", True),
            max_output_tokens=self.config.llm.get("max_output_tokens", 150),
            temperature=self.config.llm.get("temperature", 0.0),
        )
        self.rag = RAGSystem(self.retriever, self.llm, PromptBuilder(),
                             top_k=self.config.retriever["top_k"])
        self.evaluator = Evaluator()
        self.top_k = self.config.retriever["top_k"]
        self.candidate_pool = self.config.retriever.get("candidate_pool", 60)
        self.max_ops = self.config.defense["max_operations"]

        # Pre-embed the global corpus once for fast per-sample candidate selection.
        self._corpus_ids = [d["id"] for d in self.corpus]
        self._corpus_emb = self.retriever.embed([d["text"] for d in self.corpus])

    # ------------------------------------------------------------------
    # Per-sample DB construction
    # ------------------------------------------------------------------
    def _clean_docs_for_sample(self, sample: dict) -> list[dict]:
        """Top-M distractors from the corpus + the sample's gold docs."""
        q_emb = self.retriever.embed([sample["question"]])[0]
        scores = self.retriever._score(q_emb, self._corpus_emb)
        order = np.argsort(-scores)[: self.candidate_pool]
        docs = []
        seen = set()
        for i in order:
            d = self.corpus[i]
            # Source is a generic, non-revealing label (never "gold"/"poison").
            docs.append({"id": d["id"], "text": d["text"],
                         "source": d.get("source", "web")})
            seen.add(d["id"])
        # Force-include gold docs (so the correct answer is retrievable).
        # They keep a generic source - the gold role is tracked only by the judge.
        gold_ids = set(sample.get("gold_doc_ids", []))
        for d in self.corpus:
            if d["id"] in gold_ids and d["id"] not in seen:
                docs.append({"id": d["id"], "text": d["text"],
                             "source": d.get("source", "web")})
                seen.add(d["id"])
        return docs

    def _build_db(self, docs: list[dict]) -> KnowledgeDB:
        return KnowledgeDB(self.retriever, docs=docs, max_operations=self.max_ops)

    def _blackbox(self, db: KnowledgeDB):
        def _q(question: str) -> list[dict]:
            return db.search(question, self.top_k)
        return _q

    # ------------------------------------------------------------------
    # Single sample run
    # ------------------------------------------------------------------
    def run_sample(self, sample: dict, attack, defense,
                   poisoned: bool = True, verbose: bool = False) -> dict:
        clean_docs = self._clean_docs_for_sample(sample)
        db = self._build_db(clean_docs)

        poison_ids: list[str] = []
        if poisoned:
            ctx = AttackContext(
                question=sample["question"],
                attack_target=sample["attack_target"],
                sample_id=sample["id"],
                blackbox_query=self._blackbox(db),
                last_topk=[],
                max_poison_docs=self.config.attack["max_poison_docs"],
                max_bytes_per_doc=self.config.attack["max_bytes_per_doc"],
            )
            poison_texts = attack.attack(ctx) or []
            poison_texts = self._sanitize_poison(poison_texts, ctx)
            poison_ids = db.add_poison(poison_texts)
            # Let the attack see the post-injection Top-K (for logging / iterative attacks).
            ctx.last_topk = db.search(sample["question"], self.top_k)

        # Defense runs on a restricted proxy.
        defense_env = DefenseEnv(sample["question"], DefenseDBProxy(db))
        try:
            defense.defend(defense_env)
        except Exception as e:
            if verbose:
                print(f"[defense] {getattr(defense, 'name', '?')} raised {e!r}")

        # Final RAG query on the defended DB.
        out = self.rag.answer(sample["question"], db)
        response = out["response"]
        topk = out["topk"]

        ev = self.evaluator.evaluate(response, sample["attack_target"], sample["gold_answer"])
        result = {
            "sample_id": sample["id"],
            "question": sample["question"],
            "gold_answer": sample["gold_answer"],
            "attack_target": sample["attack_target"],
            "attack_name": getattr(attack, "name", type(attack).__name__),
            "defense_name": getattr(defense, "name", type(defense).__name__),
            "poisoned": poisoned,
            "response": response,
            "topk_ids": [d["id"] for d in topk],
            "poison_in_topk": sum(1 for d in topk if d.get("is_poison")),
            "poison_injected": len(poison_ids),
            "db_audit": db.audit(),
            **ev,
        }
        if verbose:
            print(f"  Q: {sample['question'][:60]} | "
                  f"atk={result['attack_name']} def={result['defense_name']} | "
                  f"resp={response[:40]!r} gold={sample['gold_answer']!r} "
                  f"target={sample['attack_target']!r} "
                  f"AS={ev['attack_success']} DS={ev['defense_success']}")
        return result

    def _sanitize_poison(self, texts: list[str], ctx: AttackContext) -> list[str]:
        """Enforce spec limits: <= max_poison_docs, each <= max_bytes_per_doc bytes."""
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        texts = texts[: ctx.max_poison_docs]
        out = []
        for t in texts:
            b = t.encode("utf-8", errors="ignore")
            if len(b) > ctx.max_bytes_per_doc:
                b = b[: ctx.max_bytes_per_doc]
                t = b.decode("utf-8", errors="ignore")
            out.append(t)
        return out

    # ------------------------------------------------------------------
    # Clean accuracy (defense on an unpoisoned DB)
    # ------------------------------------------------------------------
    def run_clean(self, sample: dict, defense) -> dict:
        res = self.run_sample(sample, attack=_NoAttack(), defense=defense, poisoned=False)
        return res

    # ------------------------------------------------------------------
    # Full round-robin matrix
    # ------------------------------------------------------------------
    def run_matrix(self, attacks: list, defenses: list,
                   splits: Optional[list[str]] = None,
                   max_samples: Optional[dict] = None,
                   measure_clean: bool = True,
                   clean_samples: int = 30,
                   verbose: bool = True) -> dict:
        splits = splits or ["public", "hidden"]
        max_samples = max_samples or {}

        all_samples = {}
        for sp in splits:
            sm = load_samples(sp)
            n = max_samples.get(sp)
            if n:
                sm = sm[:n]
            all_samples[sp] = sm

        # --- Clean accuracy per defense (no attack) ---
        clean_acc = {getattr(d, "name", type(d).__name__): [] for d in defenses}
        if measure_clean:
            clean_pool = all_samples.get("public", [])[:clean_samples]
            for defense in tqdm(defenses, desc="clean-acc"):
                dname = getattr(defense, "name", type(defense).__name__)
                for s in tqdm(clean_pool, desc=f"clean/{dname}", leave=False):
                    r = self.run_clean(s, defense)
                    clean_acc[dname].append(r["matches_gold"])

        # --- Poisoned round-robin ---
        per_pair: dict[tuple[str, str], dict] = {}
        total = sum(len(all_samples[sp]) for sp in splits) * len(attacks) * len(defenses)
        pbar = tqdm(total=total, desc="arena")
        for attack in attacks:
            aname = getattr(attack, "name", type(attack).__name__)
            for defense in defenses:
                dname = getattr(defense, "name", type(defense).__name__)
                for sp in splits:
                    for s in all_samples[sp]:
                        r = self.run_sample(s, attack, defense, poisoned=True)
                        key = (aname, dname)
                        bucket = per_pair.setdefault(key, {
                            "attack_success": [], "defense_success": [],
                            "poison_in_topk": [], "poison_injected": [],
                            "splits": {"public": 0, "hidden": 0},
                            "asr_split": {"public": [], "hidden": []},
                        })
                        bucket["attack_success"].append(r["attack_success"])
                        bucket["defense_success"].append(r["defense_success"])
                        bucket["poison_in_topk"].append(r["poison_in_topk"])
                        bucket["poison_injected"].append(r["poison_injected"])
                        bucket["splits"][sp] += 1
                        bucket["asr_split"][sp].append(r["attack_success"])
                        pbar.update(1)
        pbar.close()

        # --- Aggregate ---
        attack_names = [getattr(a, "name", type(a).__name__) for a in attacks]
        defense_names = [getattr(d, "name", type(d).__name__) for d in defenses]

        # ASR matrix [attack][defense]
        asr_matrix = {}
        robust_acc = {}  # [defense][attack] -> RobustAccuracy
        for aname in attack_names:
            asr_matrix[aname] = {}
            for dname in defense_names:
                b = per_pair.get((aname, dname))
                if not b:
                    continue
                asr = float(np.mean(b["attack_success"])) if b["attack_success"] else 0.0
                ra = float(np.mean(b["defense_success"])) if b["defense_success"] else 0.0
                asr_matrix[aname][dname] = round(asr, 4)
                robust_acc.setdefault(dname, {})[aname] = round(ra, 4)

        # Per-defense aggregate scores
        defense_scores = {}
        for dname in defense_names:
            ra_list = list(robust_acc.get(dname, {}).values())
            ra_avg = float(np.mean(ra_list)) if ra_list else 0.0
            ca = (float(np.mean(clean_acc[dname])) if clean_acc[dname] else 0.0) if measure_clean else 0.0
            defense_scores[dname] = {
                "robust_accuracy_avg": round(ra_avg, 4),
                "robust_accuracy_per_attack": robust_acc.get(dname, {}),
                "clean_accuracy": round(ca, 4),
                "defense_score": round(0.5 * ra_avg + 0.5 * ca, 4) if measure_clean else round(ra_avg, 4),
            }

        # Per-attack aggregate scores
        attack_scores = {}
        for aname in attack_names:
            asr_list = list(asr_matrix.get(aname, {}).values())
            attack_scores[aname] = {
                "asr_per_defense": asr_matrix.get(aname, {}),
                "asr_avg": round(float(np.mean(asr_list)) if asr_list else 0.0, 4),
            }

        return {
            "attacks": attack_names,
            "defenses": defense_names,
            "asr_matrix": asr_matrix,
            "attack_scores": attack_scores,
            "defense_scores": defense_scores,
            "num_samples": {sp: len(all_samples[sp]) for sp in splits},
            "retriever_backend": self.retriever.backend_name,
            "llm_model": self.llm.model_name,
        }


class _NoAttack:
    """Internal no-op attack used for CleanAccuracy measurement."""
    name = "__no_attack__"

    def attack(self, ctx: AttackContext) -> list[str]:
        return []
