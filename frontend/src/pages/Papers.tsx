import { useEffect } from 'react'
import Card from '../components/common/Card'
import Button from '../components/common/Button'
import { StatusBadge } from '../components/common/Badge'
import { usePapersStore } from '../store/papersStore'
import { papersApi } from '../api/client'
import styles from './Papers.module.css'

export default function Papers() {
  const { papers, loading, total, fetchPapers, deletePaper, retryDownload, clearAll } =
    usePapersStore()

  useEffect(() => {
    fetchPapers()
    // Poll for updates every 5 seconds
    const interval = setInterval(() => fetchPapers(), 5000)
    return () => clearInterval(interval)
  }, [fetchPapers])

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return '-'
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const stats = {
    total: papers.length,
    completed: papers.filter((p) => p.status === 'completed').length,
    pending: papers.filter((p) => p.status === 'pending' || p.status === 'running').length,
    failed: papers.filter((p) => p.status === 'failed').length,
  }

  const totalSize = papers.reduce((acc, p) => acc + (p.file_size_bytes || 0), 0)

  return (
    <div className={styles.page}>
      <Card
        title="Downloaded Papers"
        subtitle={`${total} papers from Semantic Scholar`}
        action={
          papers.length > 0 ? (
            <Button variant="ghost" size="sm" onClick={clearAll}>
              Clear All
            </Button>
          ) : null
        }
      >
        {/* Stats */}
        <div className={styles.statsGrid}>
          <div className={styles.statItem}>
            <div className={styles.statValue}>{stats.total}</div>
            <div className={styles.statLabel}>Total</div>
          </div>
          <div className={styles.statItem}>
            <div className={`${styles.statValue} ${styles.statusCompleted}`}>
              {stats.completed}
            </div>
            <div className={styles.statLabel}>Downloaded</div>
          </div>
          <div className={styles.statItem}>
            <div className={`${styles.statValue} ${styles.statusPending}`}>{stats.pending}</div>
            <div className={styles.statLabel}>In Progress</div>
          </div>
          <div className={styles.statItem}>
            <div className={styles.statValue}>{formatFileSize(totalSize)}</div>
            <div className={styles.statLabel}>Total Size</div>
          </div>
        </div>

        {/* Papers List */}
        <div className={styles.papersList}>
          {papers.length > 0 ? (
            papers.map((paper) => (
              <div key={paper.id} className={styles.paperItem}>
                <div className={styles.paperInfo}>
                  <div className={styles.paperTitle}>
                    {paper.semantic_scholar_url ? (
                      <a href={paper.semantic_scholar_url} target="_blank" rel="noopener noreferrer">
                        {paper.title}
                      </a>
                    ) : (
                      paper.title
                    )}
                  </div>
                  {paper.authors && paper.authors.length > 0 && (
                    <div className={styles.authors}>
                      {paper.authors.slice(0, 3).join(', ')}
                      {paper.authors.length > 3 && ' et al.'}
                    </div>
                  )}
                  <div className={styles.paperMeta}>
                    {paper.year && <span className={styles.metaItem}>{paper.year}</span>}
                    {paper.citation_count > 0 && (
                      <span className={styles.metaItem}>{paper.citation_count} citations</span>
                    )}
                    {paper.file_size_bytes && (
                      <span className={styles.fileSize}>{formatFileSize(paper.file_size_bytes)}</span>
                    )}
                  </div>
                  {paper.error_message && (
                    <div className={styles.errorMessage} title={paper.error_message}>
                      {paper.error_message}
                    </div>
                  )}
                </div>

                <div className={styles.paperActions}>
                  <StatusBadge status={paper.status} />

                  {paper.status === 'running' && (
                    <div className={styles.progressBar}>
                      <div
                        className={styles.progressFill}
                        style={{ width: `${paper.progress}%` }}
                      />
                    </div>
                  )}

                  <div className={styles.actionButtons}>
                    {paper.status === 'completed' && paper.local_path && (
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => window.open(papersApi.getFileUrl(paper.id), '_blank')}
                      >
                        View PDF
                      </Button>
                    )}
                    {paper.status === 'failed' && (
                      <Button variant="secondary" size="sm" onClick={() => retryDownload(paper.id)}>
                        Retry
                      </Button>
                    )}
                    <Button variant="ghost" size="sm" onClick={() => deletePaper(paper.id)}>
                      Delete
                    </Button>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className={styles.emptyList}>
              <p>No papers downloaded yet</p>
              <span className={styles.emptyHint}>
                Papers with open access PDFs will appear here when you run evaluations
              </span>
            </div>
          )}
        </div>
      </Card>
    </div>
  )
}
