import { NavLink } from 'react-router-dom'
import styles from './Sidebar.module.css'

const navItems = [
  { path: '/training', label: 'Training', icon: '⚡' },
  { path: '/evaluation', label: 'Evaluation', icon: '📊' },
  { path: '/models', label: 'Models', icon: '🧠' },
  { path: '/papers', label: 'Papers', icon: '📄' },
  { path: '/chat', label: 'Chat', icon: '💬' },
  { path: '/history', label: 'History', icon: '📜' },
]

export default function Sidebar() {
  return (
    <aside className={styles.sidebar}>
      <div className={styles.logo}>
        <div className={styles.logoIcon}>Q</div>
        <div className={styles.logoText}>
          <span className={styles.logoTitle}>Qwen Suite</span>
          <span className={styles.logoSubtitle}>Test Suite</span>
        </div>
      </div>

      <nav className={styles.nav}>
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `${styles.navItem} ${isActive ? styles.active : ''}`
            }
          >
            <span className={styles.navIcon}>{item.icon}</span>
            <span className={styles.navLabel}>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className={styles.footer}>
        <div className={styles.status}>
          <span className={styles.statusDot}></span>
          <span className={styles.statusText}>System Ready</span>
        </div>
      </div>
    </aside>
  )
}
