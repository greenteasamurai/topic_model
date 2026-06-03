interface Props {
  title: string
  text: string
}

export default function NarrativeAnalysis({ title, text }: Props) {
  const paragraphs = text.split('\n').filter(p => p.trim())

  return (
    <section className="card narrative-card">
      <h2>{title} — Narrative Analysis</h2>
      {paragraphs.map((p, i) => (
        <p key={i}>{p}</p>
      ))}
    </section>
  )
}
