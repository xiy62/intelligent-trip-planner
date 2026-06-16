import React from 'react'
import ReactDOM from 'react-dom/client'
import { ConfigProvider } from 'antd'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import 'antd/dist/reset.css'
import App from './App'
import Home from './views/Home'
import Observability from './views/Observability'
import Result from './views/Result'
import './styles.css'

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <Home /> },
      { path: 'result', element: <Result /> },
      { path: 'observability', element: <Observability /> }
    ]
  }
])

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#0f766e',
          colorInfo: '#2563eb',
          borderRadius: 16,
          fontFamily: '"Aptos", "Segoe UI", sans-serif'
        },
        components: {
          Button: {
            borderRadius: 14,
            controlHeightLG: 46
          },
          Card: {
            borderRadiusLG: 28
          },
          Input: {
            borderRadius: 14
          },
          Select: {
            borderRadius: 14
          },
          DatePicker: {
            borderRadius: 14
          }
        }
      }}
    >
      <RouterProvider router={router} />
    </ConfigProvider>
  </React.StrictMode>
)
