import { useEffect, useState } from 'react'
import Card from '../components/common/Card'
import Button from '../components/common/Button'
import Select from '../components/common/Select'
import Input from '../components/common/Input'
import Progress from '../components/common/Progress'
import { StatusBadge } from '../components/common/Badge'
import { useConfigStore } from '../store/configStore'
import { useTrainingStore } from '../store/trainingStore'
import { useTrainingWebSocket } from '../hooks/useWebSocket'
import styles from './Training.module.css'

export default function Training() {
  const { models, datasets, training, fetchConfigs } = useConfigStore()
  const {
    runs,
    currentRun,
    loading,
    fetchRuns,
    startTraining,
    cancelRun,
    updateCurrentRun,
  } = useTrainingStore()

  // Form state
  const [selectedModel, setSelectedModel] = useState('')
  const [selectedDataset, setSelectedDataset] = useState('')
  const [selectedTraining, setSelectedTraining] = useState('')
  const [overrides, setOverrides] = useState({
    learning_rate: '',
    num_train_epochs: '',
    per_device_train_batch_size: '',
  })

  // Logs state
  const [logs, setLogs] = useState<string[]>([])

  // WebSocket for real-time updates
  const { isConnected } = useTrainingWebSocket(currentRun?.id ?? null, (data) => {
    const msg = data as { type: string; data?: Record<string, unknown>; log?: string }
    if (msg.type === 'status' && msg.data) {
      updateCurrentRun(msg.data)
    }
    if (msg.type === 'log' && msg.log) {
      setLogs((prev) => [...prev.slice(-999), msg.log as string])
    }
  })

  // Load configs on mount
  useEffect(() => {
    fetchConfigs()
    fetchRuns()
  }, [fetchConfigs, fetchRuns])

  const handleStartTraining = async () => {
    if (!selectedModel || !selectedDataset || !selectedTraining) return

    const parameterOverrides: Record<string, unknown> = {}
    if (overrides.learning_rate) {
      parameterOverrides.learning_rate = parseFloat(overrides.learning_rate)
    }
    if (overrides.num_train_epochs) {
      parameterOverrides.num_train_epochs = parseInt(overrides.num_train_epochs)
    }
    if (overrides.per_device_train_batch_size) {
      parameterOverrides.per_device_train_batch_size = parseInt(overrides.per_device_train_batch_size)
    }

    setLogs([])
    await startTraining({
      model_config_name: selectedModel,
      dataset_config_name: selectedDataset,
      training_config_name: selectedTraining,
      parameter_overrides: Object.keys(parameterOverrides).length > 0 ? parameterOverrides : undefined,
    })
  }

  const modelOptions = models.map((m) => ({ value: m.name, label: m.name }))
  const datasetOptions = datasets.map((d) => ({ value: d.name, label: d.name }))
  const trainingOptions = training.map((t) => ({ value: t.name, label: t.name }))

  const isRunning = currentRun?.status === 'running'
  const progress = currentRun?.total_steps
    ? (currentRun.current_step / currentRun.total_steps) * 100
    : 0

  return (
    <div className={styles.page}>
      <div className={styles.grid}>
        {/* Configuration Panel */}
        <Card title="Configuration" subtitle="Select model, dataset, and training preset">
          <div className={styles.configForm}>
            <Select
              label="Model"
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
            <Select
              label="Training Preset"
              options={trainingOptions}
              value={selectedTraining}
              onChange={setSelectedTraining}
              placeholder="Select a preset..."
              disabled={isRunning}
            />
          </div>
        </Card>

        {/* Parameter Overrides */}
        <Card title="Parameter Overrides" subtitle="Optional training parameter adjustments">
          <div className={styles.overridesForm}>
            <Input
              label="Learning Rate"
              type="text"
              placeholder="e.g., 1e-5"
              value={overrides.learning_rate}
              onChange={(e) => setOverrides({ ...overrides, learning_rate: e.target.value })}
              disabled={isRunning}
            />
            <Input
              label="Epochs"
              type="number"
              placeholder="e.g., 3"
              value={overrides.num_train_epochs}
              onChange={(e) => setOverrides({ ...overrides, num_train_epochs: e.target.value })}
              disabled={isRunning}
            />
            <Input
              label="Batch Size"
              type="number"
              placeholder="e.g., 4"
              value={overrides.per_device_train_batch_size}
              onChange={(e) => setOverrides({ ...overrides, per_device_train_batch_size: e.target.value })}
              disabled={isRunning}
            />
          </div>

          <div className={styles.actions}>
            {isRunning ? (
              <Button variant="danger" onClick={() => currentRun && cancelRun(currentRun.id)}>
                Cancel Training
              </Button>
            ) : (
              <Button
                onClick={handleStartTraining}
                loading={loading}
                disabled={!selectedModel || !selectedDataset || !selectedTraining}
              >
                Start Training
              </Button>
            )}
          </div>
        </Card>

        {/* Training Progress */}
        <Card title="Training Progress" className={styles.progressCard}>
          {currentRun ? (
            <div className={styles.progressContent}>
              <div className={styles.progressHeader}>
                <div className={styles.runInfo}>
                  <span className={styles.runName}>{currentRun.run_name}</span>
                  <StatusBadge status={currentRun.status} />
                </div>
                <div className={styles.connectionStatus}>
                  <span className={`${styles.dot} ${isConnected ? styles.connected : ''}`}></span>
                  {isConnected ? 'Live' : 'Disconnected'}
                </div>
              </div>

              <Progress
                value={progress}
                label={`Step ${currentRun.current_step} / ${currentRun.total_steps}`}
                size="lg"
              />

              <div className={styles.metrics}>
                <div className={styles.metric}>
                  <span className={styles.metricLabel}>Epoch</span>
                  <span className={styles.metricValue}>
                    {currentRun.current_epoch.toFixed(2)} / {currentRun.total_epochs}
                  </span>
                </div>
                <div className={styles.metric}>
                  <span className={styles.metricLabel}>Current Loss</span>
                  <span className={styles.metricValue}>
                    {currentRun.current_loss?.toFixed(4) ?? 'N/A'}
                  </span>
                </div>
                <div className={styles.metric}>
                  <span className={styles.metricLabel}>Best Loss</span>
                  <span className={styles.metricValue}>
                    {currentRun.best_loss?.toFixed(4) ?? 'N/A'}
                  </span>
                </div>
              </div>

              {currentRun.wandb_run_url && (
                <a
                  href={currentRun.wandb_run_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={styles.wandbLink}
                >
                  View on WandB
                </a>
              )}
            </div>
          ) : (
            <div className={styles.emptyProgress}>
              <span>No active training run</span>
            </div>
          )}
        </Card>

        {/* Log Viewer */}
        <Card title="Training Logs" className={styles.logsCard}>
          <div className={styles.logsContainer}>
            {logs.length > 0 ? (
              logs.map((log, i) => (
                <div key={i} className={styles.logLine}>
                  {log}
                </div>
              ))
            ) : (
              <div className={styles.emptyLogs}>Logs will appear here during training...</div>
            )}
          </div>
        </Card>

        {/* Recent Runs */}
        <Card title="Recent Training Runs" className={styles.runsCard}>
          <div className={styles.runsList}>
            {runs.length > 0 ? (
              runs.slice(0, 10).map((run) => (
                <div key={run.id} className={styles.runItem}>
                  <div className={styles.runItemInfo}>
                    <span className={styles.runItemName}>{run.run_name}</span>
                    <span className={styles.runItemDate}>
                      {new Date(run.created_at).toLocaleDateString()}
                    </span>
                  </div>
                  <StatusBadge status={run.status} />
                </div>
              ))
            ) : (
              <div className={styles.emptyRuns}>No training runs yet</div>
            )}
          </div>
        </Card>
      </div>
    </div>
  )
}
