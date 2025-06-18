function drawCNN(ctx) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.font = '14px sans-serif';
  ctx.fillStyle = '#000';
  ctx.fillText('Входной слой', 10, 20);
  // Входные фильтры (3 слоя)
  for(let i=0; i<3; i++) {
    ctx.fillStyle = '#ADD8E6'; // Голубой
    ctx.fillRect(20 + i*30, 30, 20, 80);
  }
  ctx.fillStyle = '#000';
  ctx.fillText('Сверточный слой', 110, 20);
  // Сверточный слой (5 фильтров)
  for(let i=0; i<5; i++) {
    ctx.fillStyle = '#90EE90'; // Светло-зеленый
    ctx.fillRect(120 + i*25, 50, 15, 50);
  }
  ctx.fillStyle = '#000';
  ctx.fillText('Полносвязный слой', 230, 20);
  // Полносвязный слой (4 нейрона)
  for(let i=0; i<4; i++) {
    ctx.beginPath();
    ctx.arc(250 + i*30, 70, 10, 0, 2 * Math.PI);
    ctx.fillStyle = '#FFC0CB'; // Розовый
    ctx.fill();
    ctx.stroke();
  }
}

function drawTransformer(ctx) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.font = '14px sans-serif';
  ctx.fillStyle = '#000';
  ctx.fillText('Входная последовательность', 10, 20);
  // Входные токены (прямоугольники)
  for(let i=0; i<6; i++) {
    ctx.fillStyle = '#FFD700'; // Золотой
    ctx.fillRect(20 + i*40, 30, 30, 40);
    ctx.fillStyle = '#000';
    ctx.fillText('Токен', 25 + i*40, 55);
  }
  ctx.fillStyle = '#000';
  ctx.fillText('Self-Attention Модуль', 10, 90);
  // Самовнимание (3 слоя)
  for(let i=0; i<3; i++) {
    ctx.fillStyle = '#FF7F50'; // Коррал
    ctx.fillRect(20 + i*100, 100, 80, 30);
  }
  ctx.fillStyle = '#000';
  ctx.fillText('Выход', 10, 150);
  // Выходной слой
  ctx.fillStyle = '#87CEEB'; // Светло-голубой
  ctx.fillRect(20, 160, 100, 30);
  ctx.fillStyle = '#000';
  ctx.fillText('Результат', 40, 180);
}

function drawRNN(ctx) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.font = '14px sans-serif';
  ctx.fillStyle = '#000';
  ctx.fillText('Последовательность на входе', 10, 20);
  // Узлы состояния (5 кругов)
  for(let i=0; i<5; i++) {
    ctx.beginPath();
    ctx.arc(50 + i*60, 50, 20, 0, 2*Math.PI);
    ctx.fillStyle = '#98FB98'; // Бледно-зеленый
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#000';
    ctx.fillText('h' + (i+1), 45 + i*60, 55);
  }
  // Стрелки между узлами
  ctx.strokeStyle = '#000';
  for(let i=0; i<4; i++) {
    ctx.beginPath();
    ctx.moveTo(70 + i*60, 50);
    ctx.lineTo(110 + i*60, 50);
    ctx.stroke();
  }
  ctx.fillStyle = '#000';
  ctx.fillText('Выход', 10, 100);
  // Выходной узел
  ctx.beginPath();
  ctx.arc(350, 90, 20, 0, 2*Math.PI);
  ctx.fillStyle = '#FFDAB9'; // Светло-персиковый
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = '#000';
  ctx.fillText('y', 345, 95);
}

function resizeCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  return ctx;
}

function drawAll() {
  const cnnCanvas = document.getElementById('cnn-canvas');
  const transformerCanvas = document.getElementById('transformer-canvas');
  const rnnCanvas = document.getElementById('rnn-canvas');

  const cnnCtx = resizeCanvas(cnnCanvas);
  const transformerCtx = resizeCanvas(transformerCanvas);
  const rnnCtx = resizeCanvas(rnnCanvas);

  drawCNN(cnnCtx);
  drawTransformer(transformerCtx);
  drawRNN(rnnCtx);
}

window.onload = drawAll;
window.onresize = drawAll;

