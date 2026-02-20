import styles from './Badge.module.css'

interface BadgeProps {
  children: React.ReactNode
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'error' | 'info'
  size?: 'sm' | 'md'
}

export default function Badge({
  children,
  variant = 'default',
  size = 'md'
}: BadgeProps) {
  return (
    <span className={`${styles.badge} ${styles[variant]} ${styles[size]}`}>
      {children}
    </span>
  )
}

// Status-specific badge component
export function StatusBadge({ status }: { status: string }) {
  const variantMap: Record<string, 'default' | 'primary' | 'success' | 'warning' | 'error' | 'info'> = {
    pending: 'default',
    running: 'primary',
    completed: 'success',
    failed: 'error',
    cancelled: 'warning'
  }

  return (
    <Badge variant={variantMap[status] || 'default'}>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </Badge>
  )
}
