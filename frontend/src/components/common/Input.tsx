import React from 'react'
import styles from './Input.module.css'

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
  hint?: string
}

export default function Input({
  label,
  error,
  hint,
  className = '',
  ...props
}: InputProps) {
  return (
    <div className={`${styles.container} ${className}`}>
      {label && <label className={styles.label}>{label}</label>}
      <input
        className={`${styles.input} ${error ? styles.error : ''}`}
        {...props}
      />
      {hint && !error && <span className={styles.hint}>{hint}</span>}
      {error && <span className={styles.errorText}>{error}</span>}
    </div>
  )
}
