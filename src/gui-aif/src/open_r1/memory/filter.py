def should_write_memory(
    base_reward: float,
    ocr_score: float,
    consensus_score: float,
    tau_base: float = 0.8,
    tau_ocr: float = 0.7,
    tau_consensus: float = 0.6,
) -> bool:
    return (
        base_reward > tau_base
        and ocr_score > tau_ocr
        and consensus_score > tau_consensus
    )
