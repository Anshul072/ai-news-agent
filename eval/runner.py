from eval.judge import judge
from eval.sampler import sampler


def run(sqlite_store, golden_dir: str, n_golden: int = 5, n_recent: int = 5) -> list[dict]:
    """Orchestrate the full eval flow.

    Loads samples via sampler, calls judge for each, and returns a list of
    result dicts containing agent_name, identifiers, source, and scored dimensions.
    """
    samples = sampler(sqlite_store, golden_dir, n_golden, n_recent)
    results = []
    for sample in samples:
        agent_name = sample["agent_name"]
        scores = judge(agent_name, sample["inputs"], sample["output"])
        results.append(
            {
                "agent_name": agent_name,
                "article_id": sample.get("article_id"),
                "story_group_id": sample.get("story_group_id"),
                "importance_score": sample.get("importance_score"),
                "_source": sample.get("_source"),
                "scores": scores,
            }
        )
    return results
