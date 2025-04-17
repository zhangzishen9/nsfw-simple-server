const express = require('express');
const cors = require('cors');
const tf = require('@tensorflow/tfjs-node');
const nsfw = require('nsfwjs');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const os = require('os');

const app = express();
app.use(express.json());
app.use(cors());

let model = null;

// 初始化NSFW模型
async function loadModel() {
  model = await nsfw.load();
  console.log('NSFW模型加载完成');
}

// 下载图片
async function downloadImage(url) {
  const response = await axios({
    url,
    responseType: 'arraybuffer'
  });
  return Buffer.from(response.data, 'binary');
}

// 检测图片
async function detectImage(imagePath) {
  const image = await tf.node.decodeImage(await fs.promises.readFile(imagePath), 3);
  const predictions = await model.classify(image);
  image.dispose();
  return predictions;
}

app.post('/detect', async (req, res) => {
  try {
    const { imageUrl } = req.body;
    
    if (!imageUrl) {
      return res.status(400).json({ error: '请提供图片URL' });
    }

    // 确保模型已加载
    if (!model) {
      return res.status(503).json({ error: '模型尚未加载完成，请稍后重试' });
    }

    // 创建临时文件路径
    const tempDir = os.tmpdir();
    const tempFile = path.join(tempDir, `nsfw-${Date.now()}.jpg`);

    try {
      // 下载图片
      const imageBuffer = await downloadImage(imageUrl);
      await fs.promises.writeFile(tempFile, imageBuffer);

      // 检测图片
      const predictions = await detectImage(tempFile);

      // 删除临时文件
      await fs.promises.unlink(tempFile);

      res.json({
        success: true,
        predictions: predictions
      });
    } catch (error) {
      // 确保临时文件被删除
      if (fs.existsSync(tempFile)) {
        await fs.promises.unlink(tempFile);
      }
      throw error;
    }
  } catch (error) {
    console.error('检测失败:', error);
    res.status(500).json({
      success: false,
      error: '图片检测失败: ' + error.message
    });
  }
});

const PORT = process.env.PORT || 3000;

// 启动服务器前先加载模型
loadModel().then(() => {
  app.listen(PORT, () => {
    console.log(`服务器运行在 http://localhost:${PORT}`);
  });
}).catch(err => {
  console.error('模型加载失败:', err);
  process.exit(1);
});