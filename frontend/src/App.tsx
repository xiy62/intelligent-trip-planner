import { Layout } from 'antd'
import { Link, Outlet } from 'react-router-dom'

const { Header, Content, Footer } = Layout

export default function App() {
  return (
    <Layout className="app-shell">
      <Header className="app-header">
        <Link to="/" className="brand">
          Intelligent Trip Planner
        </Link>
        <nav className="app-nav">
          <Link to="/">Plan</Link>
          <Link to="/observability">Observability</Link>
        </nav>
      </Header>
      <Content className="app-content">
        <Outlet />
      </Content>
      <Footer className="app-footer">
        LangChain + LangGraph trip planning workflow
      </Footer>
    </Layout>
  )
}
