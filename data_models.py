# data_models.py

from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class Mood:
    overall_mood: str
    vader_sentiment: Dict[str, float]
    textblob_sentiment: Dict[str, float]
    emotions: Dict[str, float]

@dataclass
class Entity:
    name: str
    count: int
    entity_type: str

@dataclass
class Topic:
    id: int
    keywords: List[str]
    weight: float

@dataclass
class Chapter:
    number: int
    content: str
    mood: Mood
    entities: List[Entity]
    topics: List[Topic]

@dataclass
class Book:
    title: str
    chapters: List[Chapter]
    important_entities: List[Entity]
    themes: List[Topic]
    key_points: List[Tuple[int, str]]