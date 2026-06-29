from dataclasses import dataclass, field
from typing import TypedDict


class VaderSentiment(TypedDict):
    neg: float
    neu: float
    pos: float
    compound: float


class TextBlobSentiment(TypedDict):
    polarity: float
    subjectivity: float


@dataclass
class Mood:
    overall_mood: str
    vader_sentiment: VaderSentiment
    textblob_sentiment: TextBlobSentiment
    emotions: dict[str, float]


@dataclass
class Entity:
    name: str
    count: int
    entity_type: str


@dataclass
class Topic:
    id: int
    keywords: list[str]
    weight: float


@dataclass
class Chapter:
    number: int
    content: str
    mood: Mood
    entities: list[Entity]
    topics: list[Topic]
    arc_label: int = field(default=-1)


@dataclass
class CharacterImpact:
    name: str
    presence_avg: float
    absence_avg: float
    mood_delta: float        # absence_avg - presence_avg; positive = correlates with negative scenes
    cohens_d: float          # standardized effect size; normalizes delta by within-group spread
    lagged_delta: float      # mean sentiment change in scene i+1 after character appears in scene i
    scene_count: int
    crisis_scene_count: int


@dataclass
class ArticulationPoint:
    name: str
    components_after_removal: int
    clusters: list[list[str]]


@dataclass
class ArticulationAnalysis:
    articulation_points: list[ArticulationPoint]
    bridge_edges: list[tuple[str, str]]
    biconnected_components: list[list[str]]
    min_cooccurrence: int = field(default=1)


@dataclass
class TemporalSnapshot:
    up_to_scene: int
    articulation_point_names: list[str]
    bridge_edge_names: list[tuple[str, str]]


@dataclass
class CharacterRole:
    name: str
    role: str
    confidence: float
    reasons: list[str]


@dataclass
class Book:
    title: str
    chapters: list[Chapter]
    important_entities: list[Entity]
    themes: list[Topic]
    key_points: list[tuple[int, str]]
    character_impacts: list[CharacterImpact] = field(default_factory=list)
    character_roles: list[CharacterRole] = field(default_factory=list)
    articulation: ArticulationAnalysis | None = field(default=None)
    articulation_weighted: ArticulationAnalysis | None = field(default=None)
    articulation_temporal: list[TemporalSnapshot] = field(default_factory=list)
