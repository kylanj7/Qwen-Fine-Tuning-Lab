import { useEffect, useState } from 'react'
import Card from '../components/common/Card'
import Button from '../components/common/Button'
import Select from '../components/common/Select'
import Input from '../components/common/Input'
import Progress from '../components/common/Progress'
import { StatusBadge } from '../components/common/Badge'
import { useModelStore } from '../store/modelStore'
import styles from './Models.module.css'

const QUANTIZATION_OPTIONS = [
  { value: 'q4_k_m', label: 'Q4_K_M (Recommended)', description: '~6-8GB, good balance' },
  { value: 'q5_k_m', label: 'Q5_K_M', description: '~20% larger, higher quality' },
  { value: 'q6_k', label: 'Q6_K', description: 'Near-lossless' },
  { value: 'q8_0', label: 'Q8_0', description: 'Highest quality, largest' },
  { value: 'q3_k_m', label: 'Q3_K_M', description: 'Smaller, lower quality' },
]

export default function Models() {
  const {
    adapters,
    ggufModels,
    conversions,
    currentConversion,
    loading,
    fetchAdapters,
    fetchGGUFModels,
    fetchConversions,
    startConversion,
    deleteAdapter,
    deleteGGUFModel,
    updateCurrentConversion,
  } = useModelStore()

  // Form state
  const [selectedAdapter, setSelectedAdapter] = useState('')
  const [quantization, setQuantization] = useState('q4_k_m')
  const [outputName, setOutputName] = useState('')

  // Polling for conversion updates
  useEffect(() => {
    fetchAdapters()
    fetchGGUFModels()
    fetchConversions()

    // Poll for conversion status if running
    const interval = setInterval(() => {
      if (currentConversion?.status === 'running') {
        fetchConversions()
      }
    }, 5000)

    return () => clearInterval(interval)
  }, [fetchAdapters, fetchGGUFModels, fetchConversions, currentConversion?.status])

  const handleStartConversion = async () => {
    const adapter = adapters.find((a) => a.name === selectedAdapter)
    if (!adapter) return

    await startConversion({
      adapter_path: adapter.path,
      adapter_name: adapter.name,
      base_model: adapter.base_model,
      quantization_method: quantization,
      output_name: outputName || undefined,
    })
  }

  const handleDeleteAdapter = async (name: string) => {
    if (confirm(`Delete adapter "${name}"? This cannot be undone.`)) {
      await deleteAdapter(name)
    }
  }

  const handleDeleteGGUF = async (name: string) => {
    if (confirm(`Delete model "${name}"? This cannot be undone.`)) {
      await deleteGGUFModel(name)
    }
  }

  const adapterOptions = adapters.map((a) => ({
    value: a.name,
    label: `${a.name} (${a.size_mb.toFixed(1)} MB)`,
  }))

  const isConverting = currentConversion?.status === 'running'

  return (
    <div className={styles.page}>
      <div className={styles.grid}>
        {/* LoRA Adapters */}
        <Card title="LoRA Adapters" subtitle="Fine-tuned model adapters from training runs">
          <div className={styles.modelList}>
            {adapters.length > 0 ? (
              adapters.map((adapter) => (
                <div key={adapter.name} className={styles.modelItem}>
                  <div className={styles.modelInfo}>
                    <span className={styles.modelName}>{adapter.name}</span>
                    <div className={styles.modelMeta}>
                      {adapter.base_model && (
                        <span className={styles.metaItem}>Base: {adapter.base_model}</span>
                      )}
                      {adapter.dataset && (
                        <span className={styles.metaItem}>Dataset: {adapter.dataset}</span>
                      )}
                      <span className={styles.metaItem}>{adapter.size_mb.toFixed(1)} MB</span>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDeleteAdapter(adapter.name)}
                  >
                    Delete
                  </Button>
                </div>
              ))
            ) : (
              <div className={styles.emptyList}>
                No adapters found. Complete a training run to create adapters.
              </div>
            )}
          </div>
        </Card>

        {/* GGUF Models */}
        <Card title="GGUF Models" subtitle="Quantized models ready for inference">
          <div className={styles.modelList}>
            {ggufModels.length > 0 ? (
              ggufModels.map((model) => (
                <div key={model.name} className={styles.modelItem}>
                  <div className={styles.modelInfo}>
                    <span className={styles.modelName}>{model.name}</span>
                    <div className={styles.modelMeta}>
                      {model.quantization && (
                        <span className={styles.metaItem}>{model.quantization.toUpperCase()}</span>
                      )}
                      <span className={styles.metaItem}>{model.size_mb.toFixed(1)} MB</span>
                      {model.created_at && (
                        <span className={styles.metaItem}>
                          {new Date(model.created_at).toLocaleDateString()}
                        </span>
                      )}
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDeleteGGUF(model.name)}
                  >
                    Delete
                  </Button>
                </div>
              ))
            ) : (
              <div className={styles.emptyList}>
                No GGUF models found. Convert an adapter to create one.
              </div>
            )}
          </div>
        </Card>

        {/* Conversion Form */}
        <Card title="Convert Adapter" subtitle="Merge LoRA and convert to GGUF format">
          <div className={styles.conversionForm}>
            <Select
              label="Adapter to Convert"
              options={adapterOptions}
              value={selectedAdapter}
              onChange={setSelectedAdapter}
              placeholder="Select an adapter..."
              disabled={isConverting}
            />
            <Select
              label="Quantization Method"
              options={QUANTIZATION_OPTIONS}
              value={quantization}
              onChange={setQuantization}
              disabled={isConverting}
            />
            <Input
              label="Output Name (optional)"
              type="text"
              placeholder="Auto-generated if empty"
              value={outputName}
              onChange={(e) => setOutputName(e.target.value)}
              disabled={isConverting}
            />

            <Button
              onClick={handleStartConversion}
              loading={loading}
              disabled={!selectedAdapter || isConverting}
            >
              Start Conversion
            </Button>
          </div>
        </Card>

        {/* Conversion Progress */}
        <Card title="Conversion Progress" subtitle="Current and recent conversion jobs">
          {currentConversion ? (
            <div className={styles.conversionProgress}>
              <div className={styles.conversionHeader}>
                <span className={styles.conversionName}>{currentConversion.adapter_name}</span>
                <StatusBadge status={currentConversion.status} />
              </div>

              <Progress
                value={currentConversion.progress}
                label={currentConversion.current_stage || 'Preparing...'}
                size="lg"
              />

              <div className={styles.conversionDetails}>
                <span>Quantization: {currentConversion.quantization_method.toUpperCase()}</span>
                {currentConversion.output_path && (
                  <span>Output: {currentConversion.output_path.split('/').pop()}</span>
                )}
                {currentConversion.output_size_bytes && (
                  <span>
                    Size: {(currentConversion.output_size_bytes / (1024 * 1024 * 1024)).toFixed(2)} GB
                  </span>
                )}
              </div>

              {currentConversion.error_message && (
                <div className={styles.errorMessage}>{currentConversion.error_message}</div>
              )}
            </div>
          ) : (
            <div className={styles.emptyProgress}>No active conversion</div>
          )}

          {/* Recent Conversions */}
          {conversions.length > 0 && (
            <div className={styles.recentConversions}>
              <h4>Recent</h4>
              {conversions.slice(0, 5).map((job) => (
                <div key={job.id} className={styles.recentItem}>
                  <span className={styles.recentName}>{job.adapter_name}</span>
                  <span className={styles.recentQuant}>{job.quantization_method}</span>
                  <StatusBadge status={job.status} />
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
