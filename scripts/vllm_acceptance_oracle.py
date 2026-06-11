"""Oracle vLLM : acceptance MTP (method qwen3_5_mtp, k=4) sur les deux
régimes — continuation de corpus brut vs chat templaté — avec la métrique
native de vLLM (accepted/draft_tokens + par-position), directement
comparable à mtp_draft_acceptance_rate du bench qwen36."""

import json

from vllm import LLM, SamplingParams

MODEL = "/home/orosius/models/Qwen3.6-27B-Text-NVFP4-MTP"
CORPUS = "/home/orosius/speedoza/benches/data/bench_corpus_91k.txt"


def spec_counters(llm):
    out = {}
    for m in llm.get_metrics():
        if "spec_decode" in m.name:
            if hasattr(m, "value"):
                out[m.name] = m.value
            elif hasattr(m, "values"):
                out[m.name] = list(m.values)
    return out


def report(tag, before, after):
    def delta(name):
        a = after.get(name, 0)
        b = before.get(name, 0)
        if isinstance(a, list):
            return [x - y for x, y in zip(a, (b if isinstance(b, list) else [0] * len(a)))]
        return a - b

    drafts = delta("vllm:spec_decode_num_drafts")
    draft_tok = delta("vllm:spec_decode_num_draft_tokens")
    acc_tok = delta("vllm:spec_decode_num_accepted_tokens")
    per_pos = delta("vllm:spec_decode_num_accepted_tokens_per_pos")
    acc = acc_tok / draft_tok if draft_tok else float("nan")
    pp = [x / drafts for x in per_pos] if drafts and isinstance(per_pos, list) else per_pos
    print(f"[oracle] {tag}: drafts={drafts} draft_tokens={draft_tok} accepted={acc_tok}")
    print(f"[oracle] {tag}: ACCEPTANCE/draft = {acc:.3f}  per-pos = "
          f"{[round(x, 2) for x in pp] if isinstance(pp, list) else pp}")
    return acc


def main():
    corpus = open(CORPUS, errors="ignore").read()

    llm = LLM(
        model=MODEL,
        speculative_config={"method": "qwen3_5_mtp", "num_speculative_tokens": 4},
        max_model_len=4096,
        gpu_memory_utilization=0.88,
        enforce_eager=True,
        disable_log_stats=False,
        # Usage texte-only : saute le profiling de l'encodeur vision (c'est
        # lui qui a déclenché le JIT ninja non bridé qui a planté la machine).
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    sp = SamplingParams(temperature=0.0, max_tokens=128)

    # --- régime 1 : continuation de corpus brut (le régime du dashboard) ---
    slices = [corpus[o:o + 8000] for o in (0, 12000, 30000, 55000)]
    base = spec_counters(llm)
    llm.generate(slices, sp)
    after_raw = spec_counters(llm)
    acc_raw = report("corpus brut (4 x 8000 chars, 128 toks)", base, after_raw)

    # --- régime 2 : chat templaté (le régime des benchs serving) ---
    chats = [
        [{"role": "user", "content": corpus[:8000]}],
        [{"role": "user", "content": "Write a Python function that merges two "
                                     "sorted lists, with type hints and a docstring."}],
        [{"role": "user", "content": "Explique la différence entre un mutex et un "
                                     "sémaphore en programmation concurrente."}],
        [{"role": "user", "content": "Solve for x: 3x^2 - 12x + 9 = 0. Show each step."}],
    ]
    llm.chat(chats, sp)
    after_chat = spec_counters(llm)
    acc_chat = report("chat templaté (4 prompts, 128 toks)", after_raw, after_chat)

    print(json.dumps({"acc_raw": acc_raw, "acc_chat": acc_chat}))


if __name__ == "__main__":
    main()
