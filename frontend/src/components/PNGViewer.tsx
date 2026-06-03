interface Props {
  src: string
  title: string
}

export default function PNGViewer({ src, title }: Props) {
  return (
    <section className="card png-card">
      <h3>{title}</h3>
      <img src={src} alt={title} className="chart-img" loading="lazy" />
    </section>
  )
}
