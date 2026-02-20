import { useLocation } from 'react-router-dom'
import styles from './Header.module.css'

const pageTitles: Record<string, { title: string; description: string }> = {
  '/training': {
    title: 'Training Dashboard',
    description: 'Configure and monitor model training runs'
  },
  '/evaluation': {
    title: 'Evaluation Dashboard',
    description: 'Evaluate model performance with RAG-grounded scoring'
  },
  '/models': {
    title: 'Model Management',
    description: 'Manage LoRA adapters and GGUF models'
  },
  '/chat': {
    title: 'Chat Interface',
    description: 'Test models with interactive chat'
  },
  '/history': {
    title: 'Run History',
    description: 'View and compare past training and evaluation runs'
  }
}

export default function Header() {
  const location = useLocation()
  const pageInfo = pageTitles[location.pathname] || { title: 'Dashboard', description: '' }

  return (
    <header className={styles.header}>
      <div className={styles.titleSection}>
        <h1 className={styles.title}>{pageInfo.title}</h1>
        <p className={styles.description}>{pageInfo.description}</p>
      </div>
      <div className={styles.actions}>
        <div className={styles.indicator}>
          <span className={styles.indicatorDot}></span>
          <span className={styles.indicatorLabel}>GPU Ready</span>
        </div>
      </div>
    </header>
  )
}
