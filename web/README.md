# VoiceAPI Web (Next.js)

Modern black/orange UI for VoiceAPI with:

- Language dropdown
- Male/Female toggle
- Text input
- Style, speed, pitch, energy controls
- Custom voice cloning via speaker WAV upload
- In-browser audio playback
- WAV download button

## 1) Local run

```bash
cd web
cp .env.example .env.local
npm install
npm run dev
```

Open http://localhost:3000

## 2) Environment

Set in `.env.local`:

```bash
NEXT_PUBLIC_API_BASE=https://harshil748-voiceapi.hf.space
```

## 3) Deploy to Vercel (voiceapi.vercel.app)

1. Push repo to GitHub.
2. In Vercel, import this project and select `web` as the Root Directory.
3. Add env var:
   - `NEXT_PUBLIC_API_BASE=https://harshil748-voiceapi.hf.space`
4. Deploy.
5. In Vercel project settings, set domain to `voiceapi.vercel.app`.

## 4) API endpoints used

- Standard voice: `POST /synthesize`
- Custom clone: `POST /clone` (multipart with `speaker_wav`)
