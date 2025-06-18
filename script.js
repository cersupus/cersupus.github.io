const content = document.getElementById("content");

function showTab(tab, event) {
  document.querySelectorAll(".tab-button").forEach(btn => btn.classList.remove("active"));
  event.target.classList.add("active");

  if (tab === "training") {
    content.innerHTML = `
      <p>Обучение модели на простом примере (y = 2x - 1) — это базовый пример, где нейронная сеть учится находить зависимость между входными и выходными данными по простой линейной формуле. В данном случае, функция y = 2x - 1 — это правило, по которому для любого значения x нужно вычислить y.

Во время обучения модель получает набор примеров (вход x и правильный выход y), и пытается подстроить свои параметры так, чтобы предсказывать правильные значения как можно точнее.

Эпоха — это один проход по всему обучающему набору данных. Чем больше эпох, тем дольше модель учится и потенциально становится точнее.

Потери — это мера ошибки модели на текущем этапе обучения. Чем меньше значение потерь, тем точнее модель предсказывает результаты.</p>
      <pre id="output"></pre>
      <hr style="margin: 1rem 0;" />
      <iframe src="neron/index.html" style="width:100%; height:900px; border:none; border-radius: 12px;"></iframe>
    `;
    trainModel();
    setupDigitCanvas();
  } else if (tab === "visualization") {
    content.innerHTML = '<iframe src="visualization/visualization.html" style="width:100%; height:3600px; border:none; border-radius: 12px; @media (max-width: 600px):height:100%;"></iframe>';
  } else if (tab === "intro") {
  const intro = document.querySelector('#intro-content');
  content.innerHTML = intro ? intro.innerHTML : '<p>Секция "Введение" не найдена</p>';
  } else if (tab === "networks") {
  const intro = document.querySelector('#free-content');
  content.innerHTML = intro ? intro.innerHTML : '<p>Секция "Введение" не найдена</p>';
  }
}

    async function trainModel() {
      const output = document.getElementById("output");

      const model = tf.sequential();
      model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
      model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

      const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
      const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

      await model.fit(xs, ys, {
        epochs: 100,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            output.textContent = `Эпоха: ${epoch + 1}, Потери: ${logs.loss.toFixed(4)}`;
          }
        }
      });

      const prediction = model.predict(tf.tensor2d([5], [1, 1]));
      prediction.data().then(d => {
        output.textContent += `\nПредсказание для x=5: ${d[0].toFixed(2)}`;
      });
    }
    async function loadModel() {
  const modelUrl = 'https://tfhub.dev/tensorflow/tfjs-model/mnist_v1/1/default/1/model.json';
  return await tf.loadLayersModel(modelUrl);
}

let model_mnist = hub.Module('https://www.kaggle.com/models/tensorflow/mnist/TensorFlow1/logits/1');

function setupDigitCanvas() {
  const canvas = document.getElementById('digit-canvas');
  const ctx = canvas.getContext('2d');
  let painting = false;

  canvas.addEventListener('mousedown', () => painting = true);
  canvas.addEventListener('mouseup', () => painting = false);
  canvas.addEventListener('mouseleave', () => painting = false);
  canvas.addEventListener('mousemove', draw);

  ctx.lineWidth = 15;
  ctx.lineCap = "round";

  function draw(e) {
    if (!painting) return;
    ctx.strokeStyle = "#000";
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
    ctx.lineTo(e.offsetX + 0.1, e.offsetY + 0.1);
    ctx.stroke();
  }

  // Загрузим модель
  if (!model_mnist) {
    tf.loadLayersMode(model_mnist)
      .then(m => {
        model_mnist = m;
        console.log("MNIST модель загружена.");
      });
  }
}

function clearCanvas() {
  const canvas = document.getElementById('digit-canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById('prediction').innerText = '';
}

function predictDigit() {
  if (!model_mnist) {
    alert("Модель ещё загружается...");
    return;
  }

  const canvas = document.getElementById('digit-canvas');
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  // Создаем tf.tensor и преобразуем в 28x28 (MNIST формат)
  let tensor = tf.browser.fromPixels(imageData, 1)
    .resizeNearestNeighbor([28, 28])
    .toFloat()
    .div(255.0)
    .reshape([1, 28, 28, 1]);

  const prediction = model_mnist.predict(tensor);
  prediction.array().then(array => {
    const pred = array[0];
    const predictedDigit = pred.indexOf(Math.max(...pred));
    document.getElementById('prediction').innerText = `Предсказанная цифра: ${predictedDigit}`;
  });
}


showTab("intro", {target: document.querySelector(".tab-button.active")});

let net;
const imageElement = document.getElementById('inputImage');

async function loadModel() {
  net = await mobilenet.load();
  console.log('Модель загружена');
}

function loadImage(event) {
  const reader = new FileReader();
  reader.onload = function () {
    imageElement.src = reader.result;
  };
  reader.readAsDataURL(event.target.files[0]);
}

async function classifyImage() {
  if (!net) {
    alert("Модель ещё загружается...");
    return;
  }

  const result = await net.classify(imageElement);
  console.log(result);
  const top = result[0];

  document.getElementById('result').innerText =
    `Результат: ${top.className} (${(top.probability * 100).toFixed(2)}%)`;
}

window.onload = () => {
  loadModel();
  const firstTabButton = document.querySelector(".tab-button");
  showTab("intro", { target: firstTabButton });
};