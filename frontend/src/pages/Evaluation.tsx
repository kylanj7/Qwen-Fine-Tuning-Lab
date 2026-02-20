import { useEffect, useState, useCallback, useRef } from 'react'
import Card from '../components/common/Card'
import Button from '../components/common/Button'
import Select from '../components/common/Select'
import Input from '../components/common/Input'
import Progress from '../components/common/Progress'
import { StatusBadge } from '../components/common/Badge'
import { SingleScoreRadar } from '../components/charts/RadarChart'
import { useConfigStore } from '../store/configStore'
import { useModelStore } from '../store/modelStore'
import { useEvaluationStore } from '../store/evaluationStore'
import { useEvaluationWebSocket } from '../hooks/useWebSocket'
import { EvaluationDetailedResult, EvaluationScores, papersApi } from '../api/client'
import styles from './Evaluation.module.css'

export default function Evaluation() {
  const { datasets, fetchConfigs } = useConfigStore()
  const { ggufModels, fetchGGUFModels } = useModelStore()
  const {
    evaluations,
    currentEvaluation,
    currentResults,
    streamingResults,
    loading,
    fetchEvaluations,
    startEvaluation,
    getResults,
    cancelEvaluation,
    updateCurrentEvaluation,
    appendStreamingResult,
    clearStreamingResults,
    clearEvaluations,
    selectEvaluation,
  } = useEvaluationStore()

  // Form state
  const [selectedModel, setSelectedModel] = useState('')
  const [selectedDataset, setSelectedDataset] = useState('')
  const [sampleCount, setSampleCount] = useState('10')
  const [judgeModel, setJudgeModel] = useState('nemotron-3-nano:latest')

  // Expanded result row
  const [expandedRow, setExpandedRow] = useState<number | null>(null)

  // Live elapsed time
  const [elapsedTime, setElapsedTime] = useState<string>('')

  // Use ref to access current evaluation ID without causing callback changes
  const currentEvalIdRef = useRef<number | null>(null)
  useEffect(() => {
    currentEvalIdRef.current = currentEvaluation?.id ?? null
  }, [currentEvaluation?.id])

  // WebSocket message handler - stable callback that uses refs
  const handleWebSocketMessage = useCallback((data: unknown) => {
    const msg = data as { type: string; data?: Record<string, unknown> }

    if (msg.type === 'status' && msg.data) {
      updateCurrentEvaluation(msg.data)
    }

    if (msg.type === 'progress' && msg.data) {
      // Update progress and scores
      const progressData = msg.data as {
        current_sample?: number
        total_samples?: number
        scores?: EvaluationScores
      }
      updateCurrentEvaluation({
        current_sample: progressData.current_sample,
        total_samples: progressData.total_samples,
        scores: progressData.scores,
      })
    }

    if (msg.type === 'question_result' && msg.data) {
      // Append streaming result
      const resultData = msg.data as EvaluationDetailedResult
      appendStreamingResult(resultData)
    }

    if (msg.type === 'complete') {
      // Fetch full results on completion using ref to get current ID
      const evalId = currentEvalIdRef.current
      if (evalId) {
        getResults(evalId)
      }
      fetchEvaluations()
    }
  }, [updateCurrentEvaluation, appendStreamingResult, getResults, fetchEvaluations])

  // WebSocket for real-time updates
  const { isConnected } = useEvaluationWebSocket(
    currentEvaluation?.id ?? null,
    handleWebSocketMessage
  )

  // Fallback polling when WebSocket is disconnected
  useEffect(() => {
    const evalId = currentEvaluation?.id
    const evalStatus = currentEvaluation?.status

    if (!evalId || evalStatus !== 'running') return
    if (isConnected) return // Don't poll if WebSocket is connected

    console.log('[Evaluation] WebSocket disconnected, starting fallback polling for eval', evalId)

    // Poll every 2 seconds as fallback
    const pollInterval = setInterval(async () => {
      try {
        await useEvaluationStore.getState().getStatus(evalId)
      } catch (err) {
        console.error('[Evaluation] Polling error:', err)
      }
    }, 2000)

    return () => {
      console.log('[Evaluation] Stopping fallback polling')
      clearInterval(pollInterval)
    }
  }, [currentEvaluation?.id, currentEvaluation?.status, isConnected])

  // Load data on mount
  useEffect(() => {
    fetchConfigs()
    fetchGGUFModels()
    fetchEvaluations()
  }, [fetchConfigs, fetchGGUFModels, fetchEvaluations])

  // Live elapsed time update
  useEffect(() => {
    if (currentEvaluation?.status === 'running' && currentEvaluation?.started_at) {
      const updateElapsed = () => {
        const start = new Date(currentEvaluation.started_at!).getTime()
        const elapsed = Math.floor((Date.now() - start) / 1000)
        const mins = Math.floor(elapsed / 60)
        const secs = elapsed % 60
        setElapsedTime(`${mins}m ${secs}s`)
      }

      // Update immediately
      updateElapsed()

      // Then update every second
      const interval = setInterval(updateElapsed, 1000)
      return () => clearInterval(interval)
    } else {
      setElapsedTime('')
    }
  }, [currentEvaluation?.status, currentEvaluation?.started_at])

  const handleStartEvaluation = async () => {
    const model = ggufModels.find((m) => m.name === selectedModel)
    if (!model || !selectedDataset) return

    // Clear streaming results when starting a new evaluation
    clearStreamingResults()

    await startEvaluation({
      model_path: model.path,
      model_name: model.name,
      dataset_config_name: selectedDataset,
      sample_count: parseInt(sampleCount) || 10,
      judge_model: judgeModel,
    })
  }

  const modelOptions = ggufModels.map((m) => ({
    value: m.name,
    label: `${m.name} (${m.size_mb.toFixed(1)} MB)`,
  }))

  const datasetOptions = datasets.map((d) => ({ value: d.name, label: d.name }))

  const isRunning = currentEvaluation?.status === 'running'
  const progress = currentEvaluation?.total_samples
    ? (currentEvaluation.current_sample / currentEvaluation.total_samples) * 100
    : 0

  const scores = currentEvaluation?.scores

  return (
    <div className={styles.page}>
      <div className={styles.grid}>
        {/* Configuration Panel */}
        <Card title="Evaluation Setup" subtitle="Select model and dataset to evaluate">
          <div className={styles.configForm}>
            <Select
              label="GGUF Model"
              options={modelOptions}
              value={selectedModel}
              onChange={setSelectedModel}
              placeholder="Select a model..."
              disabled={isRunning}
            />
            <Select
              label="Dataset"
              options={datasetOptions}
              value={selectedDataset}
              onChange={setSelectedDataset}
              placeholder="Select a dataset..."
              disabled={isRunning}
            />
            <Input
              label="Sample Count"
              type="number"
              value={sampleCount}
              onChange={(e) => setSampleCount(e.target.value)}
              disabled={isRunning}
              hint="Number of test samples to evaluate"
            />
            <Input
              label="Judge Model"
              type="text"
              value={judgeModel}
              onChange={(e) => setJudgeModel(e.target.value)}
              disabled={isRunning}
              hint="Ollama model for scoring"
            />
          </div>

          <div className={styles.actions}>
            {isRunning ? (
              <Button
                variant="danger"
                onClick={() => currentEvaluation && cancelEvaluation(currentEvaluation.id)}
              >
                Cancel Evaluation
              </Button>
            ) : (
              <Button
                onClick={handleStartEvaluation}
                loading={loading}
                disabled={!selectedModel || !selectedDataset}
              >
                Start Evaluation
              </Button>
            )}
          </div>
        </Card>

        {/* Score Radar Chart */}
        <Card title="Score Overview" subtitle="3-Dimension evaluation scores">
          {scores ? (
            <div className={styles.radarContainer}>
              <SingleScoreRadar
                factualAccuracy={scores.factual_accuracy ?? 0}
                completeness={scores.completeness ?? 0}
                technicalPrecision={scores.technical_precision ?? 0}
                modelName={currentEvaluation?.model_name}
              />
              <div className={styles.overallScore}>
                <span className={styles.overallLabel}>Overall Score</span>
                <span className={styles.overallValue}>
                  {scores.overall_score?.toFixed(1) ?? 'N/A'}
                </span>
              </div>
            </div>
          ) : (
            <div className={styles.emptyRadar}>
              <span>Run an evaluation to see scores</span>
            </div>
          )}
        </Card>

        {/* Progress */}
        {currentEvaluation && (
          <Card title="Evaluation Progress" className={styles.progressCard}>
            <div className={styles.progressContent}>
              <div className={styles.progressHeader}>
                <div className={styles.evalInfo}>
                  <span className={styles.evalName}>{currentEvaluation.model_name}</span>
                  <StatusBadge status={currentEvaluation.status} />
                  {elapsedTime && (
                    <span className={styles.elapsedTime}>{elapsedTime}</span>
                  )}
                </div>
                <div className={styles.connectionStatus}>
                  <span className={`${styles.dot} ${isConnected ? styles.connected : ''}`}></span>
                  {isConnected ? 'Live' : 'Disconnected'}
                </div>
              </div>

              <Progress
                value={progress}
                label={`Sample ${currentEvaluation.current_sample} / ${currentEvaluation.total_samples}`}
                size="lg"
              />

              <div className={styles.scoreGrid}>
                <div className={styles.scoreItem}>
                  <span className={styles.scoreLabel}>Factual Accuracy</span>
                  <span className={styles.scoreValue} style={{ color: 'var(--score-factual)' }}>
                    {scores?.factual_accuracy?.toFixed(1) ?? '-'}
                  </span>
                </div>
                <div className={styles.scoreItem}>
                  <span className={styles.scoreLabel}>Completeness</span>
                  <span className={styles.scoreValue} style={{ color: 'var(--score-complete)' }}>
                    {scores?.completeness?.toFixed(1) ?? '-'}
                  </span>
                </div>
                <div className={styles.scoreItem}>
                  <span className={styles.scoreLabel}>Technical Precision</span>
                  <span className={styles.scoreValue} style={{ color: 'var(--score-precision)' }}>
                    {scores?.technical_precision?.toFixed(1) ?? '-'}
                  </span>
                </div>
              </div>
            </div>
          </Card>
        )}

        {/* Detailed Results */}
        <Card title="Detailed Results" className={styles.resultsCard}>
          {(() => {
            // Use completed results if available, otherwise show streaming results
            const resultsToShow = currentResults?.detailed_results?.length
              ? currentResults.detailed_results
              : streamingResults

            if (resultsToShow.length > 0) {
              return (
                <div className={styles.resultsTable}>
                  <div className={styles.tableHeader}>
                    <span className={styles.colIdx}>#</span>
                    <span className={styles.colQuestion}>Question</span>
                    <span className={styles.colScore}>Factual</span>
                    <span className={styles.colScore}>Complete</span>
                    <span className={styles.colScore}>Precision</span>
                    <span className={styles.colScore}>Overall</span>
                  </div>
                  {resultsToShow.map((result, idx) => {
                    const isStreaming = !currentResults?.detailed_results?.length && isRunning
                    return (
                      <div key={idx}>
                        <div
                          className={`${styles.tableRow} ${expandedRow === idx ? styles.expanded : ''} ${isStreaming && idx === resultsToShow.length - 1 ? styles.streaming : ''}`}
                          onClick={() => setExpandedRow(expandedRow === idx ? null : idx)}
                        >
                          <span className={styles.colIdx}>{result.question_idx + 1}</span>
                          <span className={styles.colQuestion}>
                            {result.question.slice(0, 60)}
                            {result.question.length > 60 ? '...' : ''}
                          </span>
                          <span className={styles.colScore}>
                            {result.scores.factual_accuracy?.toFixed(1) ?? '-'}
                          </span>
                          <span className={styles.colScore}>
                            {result.scores.completeness?.toFixed(1) ?? '-'}
                          </span>
                          <span className={styles.colScore}>
                            {result.scores.technical_precision?.toFixed(1) ?? '-'}
                          </span>
                          <span className={styles.colScore}>
                            {result.scores.overall_score?.toFixed(1) ?? '-'}
                          </span>
                        </div>
                        {expandedRow === idx && (
                          <div className={styles.expandedContent}>
                            {/* Question Section */}
                            <div className={styles.expandedSection}>
                              <h4>Question</h4>
                              <div className={styles.formattedText}>
                                {result.question.split('\n').map((paragraph, pIdx) => (
                                  <p key={pIdx}>{paragraph || '\u00A0'}</p>
                                ))}
                              </div>
                            </div>

                            {/* Model Response Section */}
                            <div className={styles.expandedSection}>
                              <h4>Model Response</h4>
                              <div className={styles.formattedText}>
                                {result.model_response.split('\n').map((paragraph, pIdx) => (
                                  <p key={pIdx}>{paragraph || '\u00A0'}</p>
                                ))}
                              </div>
                            </div>

                            {/* Judge Evaluation Section */}
                            {result.justification && (
                              <div className={styles.expandedSection}>
                                <h4>Judge Evaluation</h4>
                                <div className={styles.judgeScores}>
                                  <div className={styles.judgeScoreItem}>
                                    <span className={styles.judgeScoreLabel}>Factual Accuracy</span>
                                    <span className={styles.judgeScoreValue}>{result.scores.factual_accuracy}/100</span>
                                  </div>
                                  <div className={styles.judgeScoreItem}>
                                    <span className={styles.judgeScoreLabel}>Completeness</span>
                                    <span className={styles.judgeScoreValue}>{result.scores.completeness}/100</span>
                                  </div>
                                  <div className={styles.judgeScoreItem}>
                                    <span className={styles.judgeScoreLabel}>Technical Precision</span>
                                    <span className={styles.judgeScoreValue}>{result.scores.technical_precision}/100</span>
                                  </div>
                                </div>
                                <div className={styles.justification}>
                                  <h5>Justification</h5>
                                  <div className={styles.formattedText}>
                                    {result.justification.split('\n').map((paragraph, pIdx) => (
                                      <p key={pIdx}>{paragraph || '\u00A0'}</p>
                                    ))}
                                  </div>
                                </div>
                              </div>
                            )}

                            {/* RAG Sources Section */}
                            {result.rag_sources && result.rag_sources.length > 0 && (
                              <div className={styles.expandedSection}>
                                <h4>RAG Sources ({result.rag_sources.length} papers)</h4>
                                <div className={styles.sourcesList}>
                                  {result.rag_sources.map((source, sIdx) => {
                                    const paperId = source.url?.split('/').pop() || `paper-${sIdx}`
                                    return (
                                      <div key={sIdx} className={styles.sourceItem}>
                                        <div className={styles.sourceInfo}>
                                          <div className={styles.sourceTitle}>
                                            {source.url ? (
                                              <a href={source.url} target="_blank" rel="noopener noreferrer">
                                                {source.title}
                                              </a>
                                            ) : (
                                              source.title
                                            )}
                                          </div>
                                          <div className={styles.sourceMeta}>
                                            {source.year && <span className={styles.sourceYear}>{source.year}</span>}
                                            {source.authors && source.authors.length > 0 && (
                                              <span className={styles.sourceAuthors}>
                                                {source.authors.slice(0, 3).join(', ')}
                                                {source.authors.length > 3 && ' et al.'}
                                              </span>
                                            )}
                                            {source.is_open_access && (
                                              <span className={styles.openAccess}>Open Access</span>
                                            )}
                                          </div>
                                        </div>
                                        {source.pdf_url && (
                                          <Button
                                            variant="ghost"
                                            size="sm"
                                            onClick={(e) => {
                                              e.stopPropagation()
                                              papersApi.download({
                                                paper_id: paperId,
                                                title: source.title,
                                                pdf_url: source.pdf_url!,
                                                authors: source.authors,
                                                year: source.year,
                                                semantic_scholar_url: source.url || '',
                                                evaluation_id: currentEvaluation?.id,
                                              })
                                            }}
                                          >
                                            Download PDF
                                          </Button>
                                        )}
                                      </div>
                                    )
                                  })}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              )
            }

            return (
              <div className={styles.emptyResults}>
                <span>
                  {isRunning
                    ? 'Results will appear as questions are evaluated...'
                    : 'Detailed results will appear after evaluation completes'}
                </span>
              </div>
            )
          })()}
        </Card>

        {/* Recent Evaluations */}
        <Card
          title="Recent Evaluations"
          className={styles.historyCard}
          action={
            evaluations.length > 0 && !isRunning ? (
              <Button variant="ghost" size="sm" onClick={clearEvaluations}>
                Clear
              </Button>
            ) : null
          }
        >
          <div className={styles.evalList}>
            {evaluations.length > 0 ? (
              evaluations.slice(0, 10).map((evaluation) => (
                <div
                  key={evaluation.id}
                  className={`${styles.evalItem} ${currentEvaluation?.id === evaluation.id ? styles.selected : ''}`}
                  onClick={() => selectEvaluation(evaluation.id)}
                >
                  <div className={styles.evalItemInfo}>
                    <span className={styles.evalItemName}>{evaluation.model_name}</span>
                    <span className={styles.evalItemDataset}>{evaluation.dataset_config_name}</span>
                  </div>
                  <div className={styles.evalItemScore}>
                    {evaluation.scores?.overall_score?.toFixed(1) ?? '-'}
                  </div>
                  <StatusBadge status={evaluation.status} />
                </div>
              ))
            ) : (
              <div className={styles.emptyEvals}>No evaluations yet</div>
            )}
          </div>
        </Card>
      </div>
    </div>
  )
}
