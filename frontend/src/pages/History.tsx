import { useEffect, useState } from 'react'
import Card from '../components/common/Card'
import Button from '../components/common/Button'
import { StatusBadge } from '../components/common/Badge'
import { ComparisonRadar } from '../components/charts/RadarChart'
import { useTrainingStore } from '../store/trainingStore'
import { useEvaluationStore } from '../store/evaluationStore'
import styles from './History.module.css'

type TabType = 'training' | 'evaluation'

export default function History() {
  const [activeTab, setActiveTab] = useState<TabType>('training')
  const [selectedForComparison, setSelectedForComparison] = useState<number[]>([])

  const { runs, fetchRuns } = useTrainingStore()
  const { evaluations, fetchEvaluations, getResults, currentResults } = useEvaluationStore()

  useEffect(() => {
    fetchRuns({ limit: 50 })
    fetchEvaluations({ limit: 50 })
  }, [fetchRuns, fetchEvaluations])

  const toggleComparison = (id: number) => {
    setSelectedForComparison((prev) =>
      prev.includes(id) ? prev.filter((i) => i !== id) : prev.length < 4 ? [...prev, id] : prev
    )
  }

  const comparisonData = selectedForComparison
    .map((id) => {
      const evaluation = evaluations.find((e) => e.id === id)
      if (!evaluation?.scores) return null
      return {
        name: evaluation.model_name.slice(0, 20),
        factual_accuracy: evaluation.scores.factual_accuracy ?? 0,
        completeness: evaluation.scores.completeness ?? 0,
        technical_precision: evaluation.scores.technical_precision ?? 0,
      }
    })
    .filter(Boolean) as Array<{
    name: string
    factual_accuracy: number
    completeness: number
    technical_precision: number
  }>

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  const formatDuration = (start?: string, end?: string) => {
    if (!start || !end) return '-'
    const ms = new Date(end).getTime() - new Date(start).getTime()
    const minutes = Math.floor(ms / 60000)
    const hours = Math.floor(minutes / 60)
    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`
    }
    return `${minutes}m`
  }

  const handleExport = (type: 'training' | 'evaluation') => {
    const data = type === 'training' ? runs : evaluations
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${type}_history_${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className={styles.page}>
      {/* Tabs */}
      <div className={styles.tabs}>
        <button
          className={`${styles.tab} ${activeTab === 'training' ? styles.active : ''}`}
          onClick={() => setActiveTab('training')}
        >
          Training History
        </button>
        <button
          className={`${styles.tab} ${activeTab === 'evaluation' ? styles.active : ''}`}
          onClick={() => setActiveTab('evaluation')}
        >
          Evaluation History
        </button>
      </div>

      <div className={styles.content}>
        {activeTab === 'training' ? (
          <Card title="Training Runs" className={styles.historyCard}>
            <div className={styles.actions}>
              <Button variant="secondary" size="sm" onClick={() => handleExport('training')}>
                Export JSON
              </Button>
            </div>

            <div className={styles.table}>
              <div className={styles.tableHeader}>
                <span className={styles.colName}>Run Name</span>
                <span className={styles.colConfig}>Model</span>
                <span className={styles.colConfig}>Dataset</span>
                <span className={styles.colMetric}>Loss</span>
                <span className={styles.colTime}>Duration</span>
                <span className={styles.colStatus}>Status</span>
              </div>
              {runs.length > 0 ? (
                runs.map((run) => (
                  <div key={run.id} className={styles.tableRow}>
                    <span className={styles.colName}>
                      <span className={styles.runName}>{run.run_name}</span>
                      <span className={styles.runDate}>{formatDate(run.created_at)}</span>
                    </span>
                    <span className={styles.colConfig}>{run.model_config_name}</span>
                    <span className={styles.colConfig}>{run.dataset_config_name}</span>
                    <span className={styles.colMetric}>
                      {run.best_loss?.toFixed(4) ?? '-'}
                    </span>
                    <span className={styles.colTime}>
                      {formatDuration(run.started_at, run.completed_at)}
                    </span>
                    <span className={styles.colStatus}>
                      <StatusBadge status={run.status} />
                    </span>
                  </div>
                ))
              ) : (
                <div className={styles.emptyTable}>No training runs yet</div>
              )}
            </div>
          </Card>
        ) : (
          <>
            {/* Comparison Radar Chart */}
            {comparisonData.length > 0 && (
              <Card title="Score Comparison" className={styles.comparisonCard}>
                <ComparisonRadar evaluations={comparisonData} />
                <div className={styles.comparisonLegend}>
                  {comparisonData.map((item, idx) => (
                    <span key={idx} className={styles.legendItem}>
                      {item.name}
                    </span>
                  ))}
                </div>
              </Card>
            )}

            <Card title="Evaluation Runs" className={styles.historyCard}>
              <div className={styles.actions}>
                <span className={styles.hint}>
                  Select up to 4 evaluations to compare
                </span>
                <Button variant="secondary" size="sm" onClick={() => handleExport('evaluation')}>
                  Export JSON
                </Button>
              </div>

              <div className={styles.table}>
                <div className={styles.tableHeader}>
                  <span className={styles.colCheck}></span>
                  <span className={styles.colName}>Model</span>
                  <span className={styles.colConfig}>Dataset</span>
                  <span className={styles.colScore}>Factual</span>
                  <span className={styles.colScore}>Complete</span>
                  <span className={styles.colScore}>Precision</span>
                  <span className={styles.colScore}>Overall</span>
                  <span className={styles.colStatus}>Status</span>
                </div>
                {evaluations.length > 0 ? (
                  evaluations.map((evaluation) => (
                    <div
                      key={evaluation.id}
                      className={`${styles.tableRow} ${
                        selectedForComparison.includes(evaluation.id) ? styles.selected : ''
                      }`}
                      onClick={() => toggleComparison(evaluation.id)}
                    >
                      <span className={styles.colCheck}>
                        <input
                          type="checkbox"
                          checked={selectedForComparison.includes(evaluation.id)}
                          onChange={() => {}}
                          className={styles.checkbox}
                        />
                      </span>
                      <span className={styles.colName}>
                        <span className={styles.runName}>{evaluation.model_name}</span>
                        <span className={styles.runDate}>{formatDate(evaluation.created_at)}</span>
                      </span>
                      <span className={styles.colConfig}>{evaluation.dataset_config_name}</span>
                      <span className={styles.colScore}>
                        {evaluation.scores?.factual_accuracy?.toFixed(1) ?? '-'}
                      </span>
                      <span className={styles.colScore}>
                        {evaluation.scores?.completeness?.toFixed(1) ?? '-'}
                      </span>
                      <span className={styles.colScore}>
                        {evaluation.scores?.technical_precision?.toFixed(1) ?? '-'}
                      </span>
                      <span className={styles.colScore}>
                        <strong>{evaluation.scores?.overall_score?.toFixed(1) ?? '-'}</strong>
                      </span>
                      <span className={styles.colStatus}>
                        <StatusBadge status={evaluation.status} />
                      </span>
                    </div>
                  ))
                ) : (
                  <div className={styles.emptyTable}>No evaluations yet</div>
                )}
              </div>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}
