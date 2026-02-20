import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/common/Layout'
import Training from './pages/Training'
import Evaluation from './pages/Evaluation'
import Models from './pages/Models'
import Papers from './pages/Papers'
import Chat from './pages/Chat'
import History from './pages/History'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/training" replace />} />
        <Route path="/training" element={<Training />} />
        <Route path="/evaluation" element={<Evaluation />} />
        <Route path="/models" element={<Models />} />
        <Route path="/papers" element={<Papers />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/history" element={<History />} />
      </Routes>
    </Layout>
  )
}

export default App
