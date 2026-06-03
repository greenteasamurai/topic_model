export interface CharacterData {
  name: string
  cohens_d: number
  lagged_delta: number
  mood_delta: number
  presence_avg: number
  scene_count: number
  crisis_scene_count: number
}

export interface MoodPoint {
  scene: number
  score: number
  arc: number
  entities: string[]
  dominant_emotion: string
}

export interface Topic {
  id: number
  keywords: string[]
}

export interface AnalysisResult {
  title: string
  domain: string
  narrative_analysis: string
  characters: CharacterData[]
  mood_flow: MoodPoint[]
  topics: Topic[]
  turning_point_scene: number | null
  charts: Record<string, string>
}
