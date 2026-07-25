#!/usr/bin/env python3
"""Pre-flight: exercise every non-LLM code path of recursive_qa_agent_v4 on the subset."""
import json, random, sys
sys.path.insert(0, "/scratch/bl4363/DeepRubric-Code/data_construction")
SUB = "/scratch/bl4363/data/ASearcher-subset"

from recursive_qa_agent_v4 import normalize_url, find_page, RecursiveQAAgent, SearchNode

print("[1] loading wikilinks.json ...", flush=True)
all_links = json.load(open(f"{SUB}/wikilinks.json"))
all_urls = list(all_links.keys())
print(f"    links={len(all_urls):,}  sample_keys={list(all_links[all_urls[0]].keys())}")

print("[2] loading wiki_webpages.jsonl ...", flush=True)
pages = [json.loads(l) for l in open(f"{SUB}/wiki_webpages.jsonl", encoding="utf-8")]
pages_dict = {p["url"]: p for p in pages}
print(f"    pages={len(pages_dict):,}")

print("[3] root sampling + find_page resolution (200 draws) ...", flush=True)
random.seed(0)
miss = 0
for _ in range(200):
    u = random.choice(all_urls)
    pg = find_page(pages_dict, u)
    if pg is None or not pg.get("contents"):
        miss += 1
print(f"    unresolved roots: {miss}/200  {'OK' if miss == 0 else 'PROBLEM'}")

u = random.choice(all_urls)
entry = all_links[u]
root_query = entry["title"] if isinstance(entry, dict) and "title" in entry else u
print(f"[4] example root_url = {u}")
print(f"    root_query(before title override) = {root_query}")
print(f"    page contents[:200] = {find_page(pages_dict,u)['contents'][:200]!r}")

print("[5] tree payload builder on a synthetic tree ...", flush=True)
agent = RecursiveQAAgent(pages=pages_dict, search_client=None, max_depth=1)
root = SearchNode(query="root topic", depth=0, statements=["s1", "s2"])
root.children = [SearchNode(query=f"child {i}", depth=1, statements=[f"c{i}-stmt"]) for i in range(3)]
payload = agent.build_tree_payload(root)
print(f"    nodes={len(payload['nodes'])} leaves={len(payload['leaves'])} leaf_ids={[l['id'] for l in payload['leaves']]}")
print("PREFLIGHT_OK")
