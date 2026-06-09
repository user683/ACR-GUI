import numpy as np

from open_r1.memory import (
    GroundingMemoryBank,
    GroundingMemoryItem,
    compute_anchor_reward,
    compute_memory_reward,
)
from open_r1.memory.embedding import TextEmbeddingEncoder


def make_item(
    item_id: str,
    instruction: str,
    point: tuple[float, float],
    domain: str = "android_settings",
) -> GroundingMemoryItem:
    x, y = point
    return GroundingMemoryItem(
        id=item_id,
        domain=domain,
        instruction=instruction,
        embedding=None,
        bbox=(x - 0.02, y - 0.02, x + 0.02, y + 0.02),
        point=point,
        context_text="Search" if "search" in instruction.lower() else None,
        element_type="button",
        layout_role="top_bar",
        confidence=0.91,
        success=True,
    )


def make_bank() -> GroundingMemoryBank:
    encoder = TextEmbeddingEncoder(model_name=None, dim=64)
    return GroundingMemoryBank(encoder=encoder)


def test_add_memory_and_search_returns_relevant_item():
    bank = make_bank()
    bank.add(make_item("mem_000001", "click the search button", (0.89, 0.075)))
    bank.add(make_item("mem_000002", "open bluetooth settings", (0.30, 0.60)))

    results = bank.search("tap search button", top_k=1)

    assert len(results) == 1
    assert results[0].item.id == "mem_000001"
    assert results[0].similarity > 0


def test_save_and_load_round_trip(tmp_path):
    bank = make_bank()
    bank.add(make_item("mem_000001", "click the search button", (0.89, 0.075)))
    bank.add(make_item("mem_000002", "open bluetooth settings", (0.30, 0.60)))

    jsonl_path = tmp_path / "memory.jsonl"
    npy_path = tmp_path / "memory.npy"
    bank.save(str(jsonl_path), str(npy_path))

    loaded = make_bank()
    loaded.load(str(jsonl_path), str(npy_path))

    assert [item.id for item in loaded.items] == ["mem_000001", "mem_000002"]
    np.testing.assert_allclose(loaded.embeddings, bank.embeddings)
    assert loaded.search("search button", top_k=1)[0].item.id == "mem_000001"


def test_memory_reward_is_higher_for_nearby_points():
    bank = make_bank()
    bank.add(make_item("mem_000001", "click the search button", (0.89, 0.075)))

    near = compute_memory_reward("click search button", (0.88, 0.08), bank, top_k=1, sigma=0.15)
    far = compute_memory_reward("click search button", (0.10, 0.90), bank, top_k=1, sigma=0.15)

    assert near > far
    assert near > 0.9
    assert far < 0.01


def test_anchor_reward_combines_point_size_and_zone():
    bank = make_bank()
    bank.add(make_item("mem_000001", "click the search button", (0.89, 0.075)))

    near = compute_anchor_reward(
        "click search button",
        (0.86, 0.055, 0.90, 0.095),
        bank,
        top_k=1,
        tau_sim=0.1,
        tau_conf=0.8,
    )
    wrong_zone = compute_anchor_reward(
        "click search button",
        (0.10, 0.86, 0.14, 0.90),
        bank,
        top_k=1,
        tau_sim=0.1,
        tau_conf=0.8,
    )

    assert near > wrong_zone
    assert near > 1.0


def test_anchor_reward_respects_confidence_gate():
    bank = make_bank()
    low_conf = make_item("mem_000001", "click the search button", (0.89, 0.075))
    low_conf.confidence = 0.2
    bank.add(low_conf)

    reward = compute_anchor_reward(
        "click search button",
        (0.86, 0.055, 0.90, 0.095),
        bank,
        top_k=1,
        tau_sim=0.1,
        tau_conf=0.8,
    )

    assert reward == 0.0
