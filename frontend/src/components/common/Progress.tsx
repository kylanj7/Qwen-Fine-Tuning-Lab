import styles from './Progress.module.css'

interface ProgressProps {
  value: number
  max?: number
  label?: string
  showValue?: boolean
  size?: 'sm' | 'md' | 'lg'
  variant?: 'default' | 'success' | 'warning' | 'error'
}

export default function Progress({
  value,
  max = 100,
  label,
  showValue = true,
  size = 'md',
  variant = 'default'
}: ProgressProps) {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100)

  return (
    <div className={styles.container}>
      {(label || showValue) && (
        <div className={styles.header}>
          {label && <span className={styles.label}>{label}</span>}
          {showValue && (
            <span className={styles.value}>{Math.round(percentage)}%</span>
          )}
        </div>
      )}
      <div className={`${styles.track} ${styles[size]}`}>
        <div
          className={`${styles.bar} ${styles[variant]}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  )
}
