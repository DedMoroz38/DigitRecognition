// server.js
const express = require("express");
const bodyParser = require("body-parser");
const { spawn } = require("child_process");
const path = require("path");

const app = express();
const PORT = 3000;

app.use(express.static(path.join(__dirname, "public")));

app.use(bodyParser.json({ limit: "50mb" }));

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.post("/predict", (req, res) => {
  const imageBase64 = req.body.imageBase64;
  if (!imageBase64) {
    return res.status(400).json({ error: "No base64 image data found." });
  }

  console.log('found image data');

  const py = spawn("python3", ["predict.py"]);

  const payload = JSON.stringify({ imageBase64 });

  let pyData = "";
  py.stdout.on("data", (chunk) => {
    pyData += chunk.toString();
  });

  py.stderr.on("data", (err) => {
    console.error("Python STDERR:", err.toString());
  });

  py.on("close", (code) => {
    try {
      const result = JSON.parse(pyData);
      return res.json(result);
    } catch (parseError) {
      console.error("Error parsing Python output:", parseError);
      return res.status(500).json({ error: parseError.message });
    }
  });

  console.log('writed payload');

  py.stdin.write(payload);
  py.stdin.end();
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
