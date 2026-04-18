import { GoogleGenAI, Type, ThinkingLevel } from "@google/genai";

let aiInstance: GoogleGenAI | null = null;

function getAiClient() {
  if (!aiInstance) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY is missing. Please configure it in the Secrets panel.");
    }
    aiInstance = new GoogleGenAI({ apiKey });
  }
  return aiInstance;
}

export interface AnalysisResult {
  score: number; // 0-100, where 100 is highly likely to be a deepfake
  verdict: 'Authentic' | 'Suspicious' | 'High Risk';
  isReal: boolean;
  confidence: number;
  findings: string[];
  explanation: string;
}

export async function analyzeMedia(
  base64Data: string, 
  mimeType: string
): Promise<AnalysisResult> {
  const isVideo = mimeType.startsWith('video/');
  
  const prompt = `
    You are a world-class digital forensics expert specializing in deepfake, synthetic media, and document fraud detection.
    Analyze the provided ${isVideo ? 'video' : 'image'} for any signs of digital manipulation, AI generation, or structural artifacts.
    
    Assessment Guidelines:
    1. For Human Subjects: Detect facial inconsistencies, asymmetric pupils, boundary blurring, or "liquid" teeth artifacts common in GANs.
    2. For Documents/IDs: Identify font misalignments, digital overlays, moiré patterns, or evidence of recapture (screen-of-a-screen).
    3. For General Scenes: Check for lighting inconsistencies, unnatural textures, and edge haloing.
    
    Final Goal: Determine if this media is AUTHENTIC (Real world capture) or MANIPULATED (Synth/Deepfake/Tampered).
    Respond ONLY with a valid JSON object. Be concise in your findings and explanation.
  `;

  try {
    const ai = getAiClient();
    const response = await ai.models.generateContent({
      model: "gemini-flash-latest",
      contents: [
        {
          parts: [
            { text: prompt },
            {
              inlineData: {
                mimeType,
                data: base64Data.split(',')[1] || base64Data
              }
            }
          ]
        }
      ],
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            score: {
              type: Type.NUMBER,
              description: "Deepfake probability score from 0 to 100."
            },
            verdict: {
              type: Type.STRING,
              enum: ["Authentic", "Suspicious", "High Risk"],
              description: "The summary verdict of the analysis."
            },
            isReal: {
              type: Type.BOOLEAN,
              description: "True if specifically determined to be authentic/real, False if likely fake/manipulated."
            },
            confidence: {
              type: Type.NUMBER,
              description: "The AI's confidence in this verdict (0-100)."
            },
            findings: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "Specific detected artifacts or suspicious points."
            },
            explanation: {
              type: Type.STRING,
              description: "A detailed breakdown of the forensics analysis."
            }
          },
          required: ["score", "verdict", "isReal", "findings", "explanation", "confidence"]
        }
      }
    });

    const text = response.text || '';
    // Extract the JSON block more robustly
    let jsonStr = text;
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      jsonStr = jsonMatch[0];
    }
    
    if (!jsonStr.trim()) {
      throw new Error("The AI provided an empty analysis. This can happen with very large or complex files. Please try a smaller image/video.");
    }
    
    try {
      const result = JSON.parse(jsonStr);
      return result as AnalysisResult;
    } catch (parseError) {
      console.error("JSON Parse Error. Raw text:", text);
      throw new Error("The forensics engine returned an unreadable report format. Please try again.");
    }
  } catch (error: any) {
    console.error("Analysis failed:", error);
    throw new Error(error.message || "Failed to analyze media. Please try again.");
  }
}
