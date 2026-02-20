import {
  Radar,
  RadarChart as RechartsRadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from 'recharts'
import styles from './RadarChart.module.css'

interface ScoreData {
  name: string
  factual_accuracy?: number
  completeness?: number
  technical_precision?: number
  color?: string
}

interface RadarChartProps {
  data: ScoreData[]
  showLegend?: boolean
  size?: number
}

export default function RadarChart({ data, showLegend = true }: RadarChartProps) {
  // Transform data for Recharts
  const chartData = [
    {
      dimension: 'Factual Accuracy',
      ...Object.fromEntries(
        data.map((d) => [d.name, d.factual_accuracy ?? 0])
      ),
    },
    {
      dimension: 'Completeness',
      ...Object.fromEntries(
        data.map((d) => [d.name, d.completeness ?? 0])
      ),
    },
    {
      dimension: 'Technical Precision',
      ...Object.fromEntries(
        data.map((d) => [d.name, d.technical_precision ?? 0])
      ),
    },
  ]

  const colors = [
    'var(--score-factual)',
    'var(--score-complete)',
    'var(--score-precision)',
    'var(--score-overall)',
  ]

  return (
    <div className={styles.container}>
      <ResponsiveContainer width="100%" height={300}>
        <RechartsRadarChart data={chartData} cx="50%" cy="50%" outerRadius="70%">
          <PolarGrid stroke="var(--border-primary)" />
          <PolarAngleAxis
            dataKey="dimension"
            tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 100]}
            tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
            tickCount={5}
          />
          {data.map((item, index) => (
            <Radar
              key={item.name}
              name={item.name}
              dataKey={item.name}
              stroke={item.color || colors[index % colors.length]}
              fill={item.color || colors[index % colors.length]}
              fillOpacity={0.2}
              strokeWidth={2}
            />
          ))}
          <Tooltip
            contentStyle={{
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-primary)',
              borderRadius: 'var(--radius-lg)',
              color: 'var(--text-primary)',
            }}
            formatter={(value: number) => [`${value.toFixed(1)}%`, '']}
          />
          {showLegend && (
            <Legend
              wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)' }}
            />
          )}
        </RechartsRadarChart>
      </ResponsiveContainer>
    </div>
  )
}

// Single evaluation score display
interface SingleScoreRadarProps {
  factualAccuracy: number
  completeness: number
  technicalPrecision: number
  modelName?: string
}

export function SingleScoreRadar({
  factualAccuracy,
  completeness,
  technicalPrecision,
  modelName = 'Score',
}: SingleScoreRadarProps) {
  return (
    <RadarChart
      data={[
        {
          name: modelName,
          factual_accuracy: factualAccuracy,
          completeness: completeness,
          technical_precision: technicalPrecision,
          color: 'var(--accent-cyan)',
        },
      ]}
      showLegend={false}
    />
  )
}

// Comparison view for multiple evaluations
interface ComparisonRadarProps {
  evaluations: Array<{
    name: string
    factual_accuracy: number
    completeness: number
    technical_precision: number
    color?: string
  }>
}

export function ComparisonRadar({ evaluations }: ComparisonRadarProps) {
  return <RadarChart data={evaluations} showLegend={true} />
}
