# フロントエンド統合パターン集

## 1. リアルタイムデータ管理パターン

### 1.1 State Management Pattern

```typescript
// TypeScript/React例
interface ChartState {
  completedBars: Bar[]        // 完成バー
  currentBar: Bar | null       // 未完成バー
  indicators: {
    rci: { [period: number]: number[] }
    ema: { [period: number]: number[] }
  }
  tempValues: {               // 未完成バー用の一時値
    rci: { [period: number]: number }
    ema: { [period: number]: number }
  }
}
```

### 1.2 WebSocket Integration

```javascript
class RealtimeDataManager {
  constructor(symbol, timeframe) {
    this.ws = null
    this.calculator = new RCICalculator()
    this.subscribers = new Set()
  }

  connect() {
    this.ws = new WebSocket('wss://your-server/realtime')
    
    this.ws.onmessage = (event) => {
      const tick = JSON.parse(event.data)
      this.processTick(tick)
    }
  }

  processTick(tick) {
    // バー変換
    const bar = this.tickToBar(tick)
    
    if (bar.isComplete) {
      // 完成バーの処理
      this.handleCompletedBar(bar)
    } else {
      // 未完成バーの処理
      this.handleIncompleteBar(bar)
    }
    
    // 購読者に通知
    this.notifySubscribers()
  }
}
```

## 2. チャートコンポーネント設計

### 2.1 React Component Structure

```jsx
// ChartContainer.jsx
const ChartContainer = () => {
  const [chartData, setChartData] = useState(null)
  const [indicators, setIndicators] = useState({})
  const dataManager = useRef(null)

  useEffect(() => {
    // データマネージャーの初期化
    dataManager.current = new RealtimeDataManager(symbol, timeframe)
    
    // データ更新のサブスクライブ
    dataManager.current.subscribe((data) => {
      setChartData(data.ohlc)
      setIndicators(data.indicators)
    })

    // クリーンアップ
    return () => {
      dataManager.current?.disconnect()
    }
  }, [symbol, timeframe])

  return (
    <div className="chart-container">
      <CandlestickChart data={chartData} />
      <RCIPanel data={indicators.rci} />
      <VolumePanel data={chartData?.volume} />
    </div>
  )
}
```

### 2.2 Vue.js Composition API

```vue
<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRealtimeData } from '@/composables/useRealtimeData'

const props = defineProps({
  symbol: String,
  timeframe: String
})

const { chartData, indicators, connect, disconnect } = useRealtimeData(
  props.symbol,
  props.timeframe
)

// 未完成バーのハイライト
const currentBarHighlight = computed(() => {
  if (!chartData.value?.currentBar) return null
  return {
    x: chartData.value.currentBar.time,
    fillColor: 'rgba(255, 165, 0, 0.3)'
  }
})

onMounted(() => {
  connect()
})

onUnmounted(() => {
  disconnect()
})
</script>
```

## 3. パフォーマンス最適化パターン

### 3.1 Virtual Scrolling for Large Datasets

```javascript
class VirtualChart {
  constructor(container, totalBars) {
    this.visibleRange = { start: 0, end: 100 }
    this.totalBars = totalBars
    this.container = container
  }

  render() {
    // 表示範囲のデータのみレンダリング
    const visibleData = this.getVisibleData()
    this.renderChart(visibleData)
  }

  getVisibleData() {
    return this.data.slice(
      this.visibleRange.start,
      this.visibleRange.end
    )
  }

  onScroll(direction, amount) {
    // スクロール時に表示範囲を更新
    if (direction === 'left') {
      this.visibleRange.start = Math.max(0, this.visibleRange.start - amount)
      this.visibleRange.end = this.visibleRange.start + 100
    }
    this.render()
  }
}
```

### 3.2 Debounced Updates

```javascript
class DebouncedUpdater {
  constructor(updateFn, delay = 100) {
    this.updateFn = updateFn
    this.delay = delay
    this.timer = null
    this.pendingUpdates = []
  }

  update(data) {
    this.pendingUpdates.push(data)
    
    if (this.timer) {
      clearTimeout(this.timer)
    }
    
    this.timer = setTimeout(() => {
      this.flush()
    }, this.delay)
  }

  flush() {
    if (this.pendingUpdates.length > 0) {
      // 最新のデータのみを使用
      const latestData = this.pendingUpdates[this.pendingUpdates.length - 1]
      this.updateFn(latestData)
      this.pendingUpdates = []
    }
  }
}
```

## 4. チャートライブラリ統合

### 4.1 Plotly.js Integration

```javascript
function createRCIChart(container, data) {
  const traces = []
  
  // RCI短期
  data.rciShort.forEach((period, values) => {
    traces.push({
      x: data.timestamps,
      y: values,
      type: 'scatter',
      mode: 'lines',
      name: `RCI ${period}`,
      line: { width: 1.5 }
    })
  })

  const layout = {
    height: 250,
    margin: { t: 0, b: 30, l: 50, r: 10 },
    yaxis: {
      range: [-105, 105],
      zeroline: true,
      gridcolor: '#e0e0e0'
    },
    shapes: [
      // 買われすぎライン
      {
        type: 'line',
        y0: 80, y1: 80,
        x0: 0, x1: 1,
        xref: 'paper',
        line: { color: 'red', dash: 'dash', width: 1 }
      },
      // 売られすぎライン
      {
        type: 'line',
        y0: -80, y1: -80,
        x0: 0, x1: 1,
        xref: 'paper',
        line: { color: 'green', dash: 'dash', width: 1 }
      }
    ]
  }

  Plotly.newPlot(container, traces, layout, { responsive: true })
}
```

### 4.2 Chart.js with Streaming Plugin

```javascript
const config = {
  type: 'line',
  data: {
    datasets: [{
      label: 'RCI 9',
      borderColor: 'rgb(75, 192, 192)',
      data: []
    }]
  },
  options: {
    scales: {
      x: {
        type: 'realtime',
        realtime: {
          onRefresh: function(chart) {
            // リアルタイムデータの追加
            const rciValue = calculator.preview(currentPrice)
            chart.data.datasets[0].data.push({
              x: Date.now(),
              y: rciValue
            })
          }
        }
      },
      y: {
        min: -100,
        max: 100
      }
    }
  }
}
```

## 5. エラーハンドリングとリカバリー

### 5.1 Connection Recovery Pattern

```javascript
class ResilientConnection {
  constructor(url, options = {}) {
    this.url = url
    this.reconnectAttempts = 0
    this.maxReconnectAttempts = options.maxReconnectAttempts || 5
    this.reconnectDelay = options.reconnectDelay || 1000
  }

  connect() {
    this.ws = new WebSocket(this.url)
    
    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      this.handleError()
    }
    
    this.ws.onclose = () => {
      if (this.shouldReconnect()) {
        this.scheduleReconnect()
      }
    }
  }

  handleError() {
    // ローカルキャッシュから復元
    this.restoreFromCache()
    
    // ユーザーに通知
    this.notifyUser('Connection lost. Attempting to reconnect...')
  }

  scheduleReconnect() {
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts)
    setTimeout(() => {
      this.reconnectAttempts++
      this.connect()
    }, delay)
  }
}
```

### 5.2 Data Validation

```javascript
function validateTickData(tick) {
  const schema = {
    symbol: 'string',
    timestamp: 'number',
    bid: 'number',
    ask: 'number'
  }
  
  for (const [key, type] of Object.entries(schema)) {
    if (typeof tick[key] !== type) {
      throw new ValidationError(`Invalid ${key}: expected ${type}`)
    }
  }
  
  // 範囲チェック
  if (tick.bid <= 0 || tick.ask <= 0) {
    throw new ValidationError('Price must be positive')
  }
  
  if (tick.bid > tick.ask) {
    throw new ValidationError('Bid cannot be greater than ask')
  }
  
  return true
}
```

**重要：データ同期の検証**

```javascript
function validateDataSync(ohlcData, rciData) {
  """
  OHLCデータとRCIデータの長さが一致することを確認
  これはRCIグラフの不連続性を防ぐために重要
  """
  for (const [period, values] of Object.entries(rciData)) {
    if (ohlcData.length !== values.length) {
      console.error(`[SYNC ERROR] Period ${period}: OHLC=${ohlcData.length}, RCI=${values.length}`)
      return false
    }
  }
  return true
}

// 定期的な同期チェック
class DataSyncMonitor {
  constructor(dataManager) {
    this.dataManager = dataManager
    this.checkInterval = 5000  // 5秒ごと
    this.lastCheck = Date.now()
  }

  startMonitoring() {
    setInterval(() => {
      if (!validateDataSync(this.dataManager.ohlcData, this.dataManager.rciData)) {
        this.handleSyncError()
      }
    }, this.checkInterval)
  }

  handleSyncError() {
    console.error('Data synchronization lost')
    // 自動復旧を試みる
    this.dataManager.resyncIndicatorData()
  }
}
```

**未完成バー更新時の注意点**

```javascript
function updateIncompleteBar(price) {
  """
  未完成バーのRCI値を更新する際は、配列に追加するのではなく
  最後の値を直接更新することが重要
  """
  const previewRCI = calculator.preview(price)
  
  // 正しい実装：最後の値を更新
  if (hasIncompleteBar && rciData[period].length > 0) {
    rciData[period][rciData[period].length - 1] = previewRCI
  }
  
  // 誤った実装：値を追加してしまう（データ長の不整合を引き起こす）
  // rciData[period].push(previewRCI)  // NG!
}
```

## 6. テスト戦略

### 6.1 Unit Tests (Jest)

```javascript
describe('RCICalculator', () => {
  let calculator
  
  beforeEach(() => {
    calculator = new RCICalculator(9)
  })
  
  test('preview should not modify internal state', () => {
    const initialPrices = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
    initialPrices.forEach(p => calculator.add(p))
    
    const stateBefore = calculator.getState()
    calculator.preview(1.9)
    const stateAfter = calculator.getState()
    
    expect(stateAfter).toEqual(stateBefore)
  })
  
  test('should handle incomplete bars correctly', () => {
    // 完成バーを追加
    const completedBars = generateBars(8)
    completedBars.forEach(bar => calculator.add(bar.close))
    
    // 未完成バーのpreview
    const incompletePrice = 1.9
    const previewResult = calculator.preview(incompletePrice)
    
    // バー完成
    const finalResult = calculator.add(incompletePrice)
    
    expect(previewResult).toBeDefined()
    expect(finalResult).toBeDefined()
  })
})
```

### 6.2 E2E Tests (Cypress)

```javascript
describe('Realtime Chart', () => {
  it('should update RCI on new ticks', () => {
    cy.visit('/chart')
    
    // WebSocketモック
    cy.mockWebSocket()
    
    // 初期データ確認
    cy.get('[data-cy=rci-value]').should('exist')
    
    // ティック送信
    cy.sendMockTick({ price: 1.1005 })
    
    // RCI更新確認
    cy.get('[data-cy=rci-value]')
      .should('not.equal', initialValue)
  })
})
```

## 7. デプロイメント設定

### 7.1 Docker Configuration

```dockerfile
# Frontend Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### 7.2 Environment Configuration

```javascript
// config.js
const config = {
  development: {
    wsUrl: 'ws://localhost:8080',
    apiUrl: 'http://localhost:3000',
    updateInterval: 1000
  },
  production: {
    wsUrl: 'wss://api.yourservice.com',
    apiUrl: 'https://api.yourservice.com',
    updateInterval: 500
  }
}

export default config[process.env.NODE_ENV || 'development']
```

## まとめ

これらのパターンを組み合わせることで、高性能でスケーラブルなリアルタイムチャートアプリケーションを構築できます。重要なのは：

1. **状態管理の明確化**: 完成/未完成データの分離
2. **パフォーマンス最適化**: 仮想スクロール、デバウンス
3. **エラーハンドリング**: 自動リカバリー、データ検証
4. **テスト**: ユニット・E2Eテストの充実

---

*Last Updated: 2025-01-26*
*Version: 1.1.0*