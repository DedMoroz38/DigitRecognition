const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");


ctx.fillStyle = "#1d1e24";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;

function startDraw(e) {
  e.preventDefault();
  drawing = true;
  draw(e);
}

function endDraw() {
  drawing = false;
  sendImageToServer();
}

function draw(e) {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  ctx.lineWidth = 30;
  ctx.lineCap = "round";
  ctx.strokeStyle = "white";
  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function clearCanvas() {
  ctx.fillStyle = "#1d1e24";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mouseup", endDraw);
// canvas.addEventListener("mouseout", endDraw);
canvas.addEventListener("mousemove", draw);

function createBWCanvas() {
  const tempCanvas = document.createElement("canvas");
  const tempCtx = tempCanvas.getContext("2d");

  tempCanvas.width = canvas.width;
  tempCanvas.height = canvas.height;

  tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

  tempCtx.drawImage(canvas, 0, 0);

  const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
  const data = imageData.data;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];     
    const g = data[i + 1]; 
    const b = data[i + 2]; 

    if (r > 200 && g > 200 && b > 200) { 
      data[i] = 0;    
      data[i + 1] = 0; 
      data[i + 2] = 0; 
    } else {
      data[i] = 255;    
      data[i + 1] = 255; 
      data[i + 2] = 255; 
    }
  }

  tempCtx.putImageData(imageData, 0, 0);

  return tempCanvas; 
}

function insertOldImage() {
  const oldImageContainer = document.getElementById("oldImageContainer");

  oldImageContainer.innerHTML = "";

  const oldImageURL = canvas.toDataURL("image/png");

  const imgElement = document.createElement("img");
  imgElement.src = oldImageURL;

  oldImageContainer.appendChild(imgElement);
}

function sendImageToServer() {
  const blur = document.getElementById("blur");
  const bwCanvas = createBWCanvas();
  const dataURL = bwCanvas.toDataURL("image/png");

  blur.style.display = "block";

  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ imageBase64: dataURL })
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      document.getElementById("digit").innerHTML = 
        `<span style="color:red;">Error: ${data.error}</span>`;
    } else {
      document.getElementById("digit").innerText = data.digit
      document.getElementById("confidence").innerText = data.confidence
    }
  })
  .catch(err => {
    document.getElementById("digit").innerHTML = 
      `<span style="color:red;">Error: ${err}</span>`;
  }).finally(() => {
    blur.style.display = "none";
    insertOldImage();
    clearCanvas();
  });
}