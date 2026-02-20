import React from 'react'
import styles from './Card.module.css'

interface CardProps {
  children: React.ReactNode
  title?: string
  subtitle?: string
  action?: React.ReactNode
  className?: string
  padding?: 'none' | 'sm' | 'md' | 'lg'
  hover?: boolean
}

export default function Card({
  children,
  title,
  subtitle,
  action,
  className = '',
  padding = 'md',
  hover = false
}: CardProps) {
  return (
    <div className={`${styles.card} ${styles[padding]} ${hover ? styles.hover : ''} ${className}`}>
      {(title || subtitle || action) && (
        <div className={styles.header}>
          <div className={styles.headerText}>
            {title && <h3 className={styles.title}>{title}</h3>}
            {subtitle && <p className={styles.subtitle}>{subtitle}</p>}
          </div>
          {action && <div className={styles.headerAction}>{action}</div>}
        </div>
      )}
      <div className={styles.content}>
        {children}
      </div>
    </div>
  )
}
