import express, { Request, Response } from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import multer from 'multer';

import Anthropic from '@anthropic-ai/sdk';

dotenv.config();

const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
});

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Multer for image upload
const upload = multer({ storage: multer.memoryStorage() });

// Health check
app.get('/api/health', (req: Request, res: Response) => {
  res.json({ status: 'ok' });
});

// Plant analysis endpoint
app.post('/api/analyze', upload.single('image'), async (req: Request, res: Response) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image provided' });
    }
    const base64Image = req.file.buffer.toString('base64');
    const huggingFaceResponse = await fetch(
        'https://api-inference.huggingface.co/models/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification',
        {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${process.env.HUGGING_FACE_TOKEN}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ inputs: base64Image }),
        }
    );
    const huggingFaceData = await huggingFaceResponse.json();

    const topResult = huggingFaceData[0];

    const message = await anthropic.messages.create({
  model: 'claude-sonnet-4-20250514',
  max_tokens: 1000,
  messages: [{
        role: 'user',
        content: `You are a plant health expert. A plant image was classified as:
        **Classification:** ${topResult.label}
        **Confidence:** ${(topResult.score * 100).toFixed(1)}%

        Based on this classification, provide:

        1. **Health Status:** Is the plant healthy or sick? If sick, what's the disease/deficiency?

        2. **Care Tips:** Provide 3-5 specific, actionable recommendations:
        - Watering frequency and amount
        - Sunlight requirements
        - Soil/nutrient needs
        - Pest/disease control (if applicable)
        - Any other critical care steps

        3. **Maturity Estimate:** When will this plant reach healthy adult growth stage? Provide a date estimate from today (${new Date().toLocaleDateString()}) based on:
        - Current health status
        - Typical growth cycle for this species
        - Recovery time if diseased

        Format your response clearly with headers. Be concise but thorough.`
    }]
    });
    const claudeResponse = message.content[0].type === 'text' ? message.content[0].text: '';

    res.json({
        classification: topResult,
        advice: claudeResponse,
    });

  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});