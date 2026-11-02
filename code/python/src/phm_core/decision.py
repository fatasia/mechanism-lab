"""每个部件分别持有状态；模型本身不存储告警状态。"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DecisionConfig:
    enter: float
    exit: float
    required: int = 3
    window: int = 5
    recovery: int = 3

    def __post_init__(self) -> None:
        if not (
            self.exit < self.enter
            and 1 <= self.required <= self.window
            and self.recovery > 0
        ):
            raise ValueError(
                "告警参数须满足 exit < enter、1 <= required <= window、recovery > 0"
            )


@dataclass
class DecisionState:
    hits: list[bool] = field(default_factory=list)
    active: bool = False
    recovery_count: int = 0


def step(
    score: float | None, state: DecisionState, config: DecisionConfig
) -> tuple[str, DecisionState]:
    """None 表示运行段中断，复位状态；返回事件和新的状态。"""
    if score is None:
        return "reset", DecisionState()
    hits = (state.hits + [score > config.enter])[-config.window :]
    recovering = state.recovery_count + 1 if score < config.exit else 0
    if state.active and recovering >= config.recovery:
        return "closed", DecisionState()
    if not state.active and sum(hits) >= config.required:
        return "opened", DecisionState(hits, True, 0)
    return "none", DecisionState(hits, state.active, recovering if state.active else 0)
