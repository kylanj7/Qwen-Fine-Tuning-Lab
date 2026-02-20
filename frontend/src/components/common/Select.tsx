import React from 'react'
import styles from './Select.module.css'

interface Option {
  value: string
  label: string
  description?: string
}

interface SelectProps {
  label?: string
  options: Option[]
  value: string
  onChange: (value: string) => void
  placeholder?: string
  disabled?: boolean
  className?: string
}

export default function Select({
  label,
  options,
  value,
  onChange,
  placeholder = 'Select...',
  disabled = false,
  className = ''
}: SelectProps) {
  return (
    <div className={`${styles.container} ${className}`}>
      {label && <label className={styles.label}>{label}</label>}
      <select
        className={styles.select}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
      >
        <option value="" disabled>{placeholder}</option>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  )
}
