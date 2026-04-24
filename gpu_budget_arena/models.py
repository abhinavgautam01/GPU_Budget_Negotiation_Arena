from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Difficulty = Literal["easy", "medium", "hard"]
TaskType = Literal["single_trade", "market_round", "coalition_market"]
ActionType = Literal[
    "send_offer",
    "accept_offer",
    "reject_offer",
    "counter_offer",
    "reserve_capacity",
    "release_capacity",
    "form_coalition",
    "commit_to_coalition",
    "allocate_to_job",
    "send_message",
    "wait",
    "finish",
]


class ResetConfig(BaseModel):
    task_type: TaskType = "single_trade"
    difficulty: Difficulty | None = None
    seed: int = 42


class GpuNegotiationAction(BaseModel):
    action_type: ActionType
    target_lab_id: str | None = None
    offer_id: str | None = None
    coalition_id: str | None = None
    block_ids: list[str] | None = None
    requested_block_ids: list[str] | None = None
    job_id: str | None = None
    payment: float | None = None
    message: str | None = None
    conditions: dict[str, object] | None = None

    @field_validator("message")
    @classmethod
    def cap_message(cls, value: str | None) -> str | None:
        if value is not None and len(value) > 500:
            raise ValueError("message must be 500 characters or fewer")
        return value


class JobView(BaseModel):
    job_id: str
    gpu_hours_required: float
    deadline_round: int
    base_value: float
    urgency_multiplier: float
    min_reliability: float
    partial_credit_allowed: bool
    private_notes: str
    completed: bool = False
    allocated_block_ids: list[str] = Field(default_factory=list)


class GpuBlockView(BaseModel):
    block_id: str
    start_round: int
    duration: int
    gpu_count: int
    reliability: float
    energy_cost: float
    owner_lab_id: str
    reserved_by: str | None = None
    allocated_to_job_id: str | None = None
    status: Literal["available", "reserved", "committed", "used", "failed"]

    @property
    def gpu_hours(self) -> float:
        return float(self.duration * self.gpu_count)


class VisibleLabView(BaseModel):
    lab_id: str
    public_name: str
    reputation: float
    public_budget_band: Literal["low", "medium", "high"]
    public_demand: Literal["low", "medium", "high"]
    owned_block_ids: list[str]
    active_offer_count: int
    coalition_ids: list[str]


class OfferView(BaseModel):
    offer_id: str
    from_lab_id: str
    to_lab_id: str
    round_created: int
    expires_round: int
    offered_blocks: list[str] = Field(default_factory=list)
    requested_blocks: list[str] = Field(default_factory=list)
    payment: float = 0.0
    message: str | None = None
    conditions: dict[str, object] = Field(default_factory=dict)
    status: Literal["active", "accepted", "rejected", "expired", "invalid"] = "active"
    linked_offer_id: str | None = None


class CoalitionView(BaseModel):
    coalition_id: str
    members: list[str]
    purpose: str
    commitments: dict[str, list[str]] = Field(default_factory=dict)
    trust_score: float = 0.5
    round_created: int
    expires_round: int
    breach_status: Literal["none", "breached", "excused"] = "none"


class MessageView(BaseModel):
    from_lab_id: str
    to_lab_id: str | None = None
    round_index: int
    message: str
    signal: Literal["urgency", "fairness", "coalition", "threat", "neutral"] = "neutral"


class PublicMarketView(BaseModel):
    available_blocks: list[GpuBlockView]
    market_pressure: Literal["low", "medium", "high"]
    shock_history: list[str] = Field(default_factory=list)


class ActionResult(BaseModel):
    ok: bool
    code: str
    message: str


class RewardBreakdown(BaseModel):
    job_utility_score: float = 0.0
    deal_quality_score: float = 0.0
    coalition_reliability_score: float = 0.0
    budget_efficiency_score: float = 0.0
    negotiation_efficiency_score: float = 0.0
    market_adaptation_score: float = 0.0
    invalid_action_penalty: float = 0.0
    spam_penalty: float = 0.0
    breach_penalty: float = 0.0
    normalized_reward: float = 0.0


class GpuNegotiationObservation(BaseModel):
    task_id: str
    difficulty: Difficulty
    round_index: int
    max_rounds: int
    controlled_lab_id: str
    controlled_lab_budget: float
    controlled_lab_reputation: float
    private_jobs: list[JobView]
    owned_blocks: list[GpuBlockView]
    public_market: PublicMarketView
    visible_labs: list[VisibleLabView]
    active_offers: list[OfferView]
    active_coalitions: list[CoalitionView]
    message_history: list[MessageView]
    last_action_result: ActionResult | None
    reward: float
    cumulative_reward: float
    reward_breakdown: RewardBreakdown
    done: bool


class LabState(BaseModel):
    lab_id: str
    public_name: str
    budget: float
    reputation: float = 0.5
    owned_blocks: list[str] = Field(default_factory=list)
    private_jobs: list[JobView] = Field(default_factory=list)
    opponent_archetype: str | None = None


class EnvironmentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    task_type: TaskType
    difficulty: Difficulty
    seed: int
    round_index: int
    max_rounds: int
    controlled_lab_id: str
    labs: dict[str, LabState]
    blocks: dict[str, GpuBlockView]
    offers: dict[str, OfferView]
    coalitions: dict[str, CoalitionView]
    messages: list[MessageView]
    shock_schedule: dict[int, str]
    shock_history: list[str]
    last_action_result: ActionResult | None = None
    last_reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    cumulative_reward: float = 0.0
    done: bool = False
    action_fingerprints: dict[str, int] = Field(default_factory=dict)
    completed_job_values: dict[str, float] = Field(default_factory=dict)

