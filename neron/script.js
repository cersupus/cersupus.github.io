// Получаем кнопки вкладок и содержимое вкладок
const tabButtons = document.querySelectorAll('.tab-btn')
const tabContents = document.querySelectorAll('.tab-content')

// Назначаем обработчики на каждую кнопку вкладки
tabButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab
    tabContents.forEach(tc => {
      tc.classList.toggle('active', tc.id === target)
    })
  })
})

// === 1. Классификация Собака/Кошка ===
// Получаем элементы DOM для загрузки изображения, кнопки и вывода результата
const classifyUpload = document.getElementById('classifyUpload')
const classifyImg = document.getElementById('classifyImg')
const classifyBtn = document.getElementById('classifyBtn')
const classifyResult = document.getElementById('classifyResult')

// Переменная для хранения загруженной модели
let classifyModel = null

// URL модели MobileNet
const MODEL_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'

// Загружаем модель при старте
async function loadClassifyModel() {
  classifyResult.textContent = 'Загрузка модели...'
  classifyModel = await tf.loadLayersModel(MODEL_URL)
  classifyResult.textContent = 'Модель загружена. Загрузите изображение.'
}
loadClassifyModel()

// Обработчик загрузки изображения
classifyUpload.addEventListener('change', () => {
  const file = classifyUpload.files[0]
  if (!file) return
  const url = URL.createObjectURL(file)
  classifyImg.src = url
  classifyResult.textContent = ''
})

// Обработчик нажатия на кнопку "Распознать"
classifyBtn.addEventListener('click', async () => {
  if (!classifyModel) {
    classifyResult.textContent = 'Модель не загружена'
    return
  }
  if (!classifyImg.src) {
    classifyResult.textContent = 'Загрузите изображение'
    return
  }

  const img = document.createElement('img')
  img.crossOrigin = 'anonymous'
  img.src = classifyImg.src

  // Обработка изображения при его загрузке
  img.onload = async () => {
    let tensor = tf.browser.fromPixels(img).resizeNearestNeighbor([224, 224]).toFloat()
    tensor = tensor.div(255).expandDims()  // Нормализация

    const predictions = await classifyModel.predict(tensor).data()

    // Индексы классов для кошек и собак в ImageNet
    const catIndices = [281, 282, 283, 284, 285]
    const dogIndices = [151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,
      170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,
      189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,
      208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,
      227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,
      246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,
      265,266,267,268
    ]

    // Суммируем вероятности
    let catProb = 0
    for (const i of catIndices) catProb += predictions[i]
    let dogProb = 0
    for (const i of dogIndices) dogProb += predictions[i]

    // Определяем итоговую метку
    let label, confidence
    if (catProb > dogProb) {
      label = 'Кошка'
      confidence = catProb
    } else {
      label = 'Собака'
      confidence = dogProb
    }

    classifyResult.textContent = `Распознано: ${label} (уверенность ${(confidence*100).toFixed(2)}%)`
  }
})

// === 2. Сегментация BodyPix ===
// Получаем DOM-элементы
const segmentUpload = document.getElementById('segmentUpload')
const segImageCanvas = document.getElementById('segImageCanvas')
const segMaskCanvas = document.getElementById('segMaskCanvas')
const segmentBtn = document.getElementById('segmentBtn')

const segImageCtx = segImageCanvas.getContext('2d')
const segMaskCtx = segMaskCanvas.getContext('2d')

let bodyPixModel = null
let segImageElement = new Image()

// Загрузка изображения для сегментации
segmentUpload.addEventListener('change', (e) => {
  const file = e.target.files[0]
  if (!file) return
  const reader = new FileReader()
  reader.onload = () => {
    segImageElement.onload = () => {
      segImageCtx.clearRect(0, 0, segImageCanvas.width, segImageCanvas.height)
      segMaskCtx.clearRect(0, 0, segMaskCanvas.width, segMaskCanvas.height)
      segImageCtx.drawImage(segImageElement, 0, 0, segImageCanvas.width, segImageCanvas.height)
    }
    segImageElement.src = reader.result
  }
  reader.readAsDataURL(file)
})

// Кнопка сегментации
segmentBtn.addEventListener('click', async () => {
  if (!segImageElement.src) {
    alert('Сначала загрузите изображение!')
    return
  }
  if (!bodyPixModel) {
    bodyPixModel = await bodyPix.load()
  }

  const segmentation = await bodyPixModel.segmentPerson(segImageCanvas, {
    internalResolution: 'medium',
    segmentationThreshold: 0.7
  })

  const mask = bodyPix.toMask(segmentation)

  bodyPix.drawMask(segMaskCanvas, segImageCanvas, mask, {
    opacity: 0.8,
    maskBlurAmount: 0,
    flipHorizontal: false
  })
})

// === 3. Крестики-нолики ===
// DOM-элементы
const boardDiv = document.getElementById('ticTacToeBoard')
const statusP = document.getElementById('ticTacToeStatus')
const resetBtn = document.getElementById('ticTacToeResetBtn')

// Состояние игры
let boardState = Array(9).fill(null)
let playerTurn = true
let gameOver = false

// Отрисовка доски
function renderBoard() {
  boardDiv.innerHTML = ''
  boardState.forEach((cell, i) => {
    const cellDiv = document.createElement('div')
    cellDiv.classList.add('cell')
    if (cell !== null) {
      cellDiv.textContent = cell
      cellDiv.classList.add('disabled')
    }
    if (gameOver) {
      cellDiv.classList.add('disabled')
    }
    cellDiv.addEventListener('click', () => playerMove(i))
    boardDiv.appendChild(cellDiv)
  })
  updateStatus()
}

// Обновление текста статуса
function updateStatus() {
  if (gameOver) return
  statusP.textContent = playerTurn ? 'Ход игрока (X)' : 'Ход ИИ (O)'
}

// Проверка победителя
function checkWin(b, player) {
  const wins = [
    [0,1,2], [3,4,5], [6,7,8],
    [0,3,6], [1,4,7], [2,5,8],
    [0,4,8], [2,4,6]
  ]
  return wins.some(line => line.every(i => b[i] === player))
}

// Проверка на ничью
function checkDraw(b) {
  return b.every(cell => cell !== null)
}

// Ход игрока
function playerMove(i) {
  if (!playerTurn || gameOver || boardState[i] !== null) return
  boardState[i] = 'X'
  playerTurn = false
  renderBoard()
  if (checkWin(boardState, 'X')) {
    statusP.textContent = 'Игрок выиграл!'
    gameOver = true
    return
  }
  if (checkDraw(boardState)) {
    statusP.textContent = 'Ничья!'
    gameOver = true
    return
  }
  setTimeout(aiMove, 500)
}

// Ход ИИ (рандомно)
function aiMove() {
  if (gameOver) return
  let emptyIndices = boardState.map((v,i) => v === null ? i : null).filter(i => i !== null)
  if (emptyIndices.length === 0) return
  const choice = emptyIndices[Math.floor(Math.random() * emptyIndices.length)]
  boardState[choice] = 'O'
  playerTurn = true
  renderBoard()
  if (checkWin(boardState, 'O')) {
    statusP.textContent = 'ИИ выиграл!'
    gameOver = true
    return
  }
  if (checkDraw(boardState)) {
    statusP.textContent = 'Ничья!'
    gameOver = true
    return
  }
}

// Сброс игры
resetBtn.addEventListener('click', () => {
  boardState = Array(9).fill(null)
  playerTurn = true
  gameOver = false
  renderBoard()
})

renderBoard()  // Инициализация при запуске
