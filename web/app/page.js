"use client";

import { useMemo, useState } from "react";

const LANGUAGES = [
	"hindi",
	"bengali",
	"marathi",
	"telugu",
	"kannada",
	"bhojpuri",
	"chhattisgarhi",
	"maithili",
	"magahi",
	"english",
	"gujarati",
];

const STYLE_PRESETS = [
	"default",
	"calm",
	"happy",
	"sad",
	"slow",
	"fast",
	"soft",
	"loud",
	"excited",
];

const LANG_VOICE_MAP = {
	hindi: { male: "hi_male", female: "hi_female" },
	bengali: { male: "bn_male", female: "bn_female" },
	marathi: { male: "mr_male", female: "mr_female" },
	telugu: { male: "te_male", female: "te_female" },
	kannada: { male: "kn_male", female: "kn_female" },
	bhojpuri: { male: "bho_male", female: "bho_female" },
	chhattisgarhi: { male: "hne_male", female: "hne_female" },
	maithili: { male: "mai_male", female: "mai_female" },
	magahi: { male: "mag_male", female: "mag_female" },
	english: { male: "en_male", female: "en_female" },
	gujarati: { male: "gu_mms", female: "gu_mms" },
};

const CLONE_SCRIPT = {
	hindi: [
		"नमस्ते, मेरा नाम [अपना नाम] है और मैं आज अपनी आवाज़ रिकॉर्ड कर रहा/रही हूँ।",
		"कृपया मातृ स्वास्थ्य से जुड़ी जानकारी ध्यान से सुनें और समय पर जांच कराएं।",
		"मैं साफ़ आवाज़ में धीरे और समान गति से बोल रहा/रही हूँ।",
	],
	bengali: [
		"নমস্কার, আমার নাম [আপনার নাম] এবং আমি আজ আমার কণ্ঠ রেকর্ড করছি।",
		"মাতৃস্বাস্থ্য সংক্রান্ত তথ্য মনোযোগ দিয়ে শুনুন এবং সময়মতো পরীক্ষা করুন।",
		"আমি স্পষ্টভাবে এবং একই গতিতে কথা বলছি।",
	],
	marathi: [
		"नमस्कार, माझे नाव [तुमचे नाव] आहे आणि मी आज माझा आवाज रेकॉर्ड करत आहे.",
		"मातृ आरोग्याशी संबंधित माहिती लक्षपूर्वक ऐका आणि वेळेवर तपासणी करा.",
		"मी स्पष्ट आणि स्थिर गतीने बोलत आहे.",
	],
	telugu: [
		"నమస్కారం, నా పేరు [మీ పేరు] మరియు నేను ఈ రోజు నా స్వరాన్ని రికార్డ్ చేస్తున్నాను.",
		"మాతృ ఆరోగ్య సమాచారాన్ని జాగ్రత్తగా విని సమయానికి పరీక్ష చేయించుకోండి.",
		"నేను స్పష్టంగా మరియు సమాన వేగంతో మాట్లాడుతున్నాను.",
	],
	kannada: [
		"ನಮಸ್ಕಾರ, ನನ್ನ ಹೆಸರು [ನಿಮ್ಮ ಹೆಸರು] ಮತ್ತು ನಾನು ಇಂದು ನನ್ನ ಧ್ವನಿಯನ್ನು ದಾಖಲಿಸುತ್ತಿದ್ದೇನೆ.",
		"ತಾಯಿ ಆರೋಗ್ಯದ ಮಾಹಿತಿಯನ್ನು ಗಮನದಿಂದ ಕೇಳಿ ಮತ್ತು ಸಮಯಕ್ಕೆ ತಪಾಸಣೆ ಮಾಡಿಸಿಕೊಳ್ಳಿ.",
		"ನಾನು ಸ್ಪಷ್ಟವಾಗಿ ಮತ್ತು ಒಂದೇ ವೇಗದಲ್ಲಿ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ.",
	],
	bhojpuri: [
		"प्रणाम, हमार नाम [आपन नाव] बा आ आज हम आपन आवाज रिकॉर्ड करत बानी।",
		"मातृ स्वास्थ्य से जुड़ल जानकारी ध्यान से सुनीं आ समय पर जांच कराईं।",
		"हम साफ आवाज में धीरे-धीरे बोलत बानी।",
	],
	chhattisgarhi: [
		"नमस्कार, मोर नाव [आपके नाव] हे अउ मंय आज अपन आवाज रिकॉर्ड करत हंव।",
		"मातृ स्वास्थ्य के जानकारी ला ध्यान ले सुनव अउ समय म जांच करवावव।",
		"मंय साफ अउ एके जइसन गति म बोलत हंव।",
	],
	maithili: [
		"प्रणाम, हमर नाम [अपन नाम] अछि आ आइ हम अपन आवाज रिकॉर्ड कऽ रहल छी।",
		"मातृ स्वास्थ्यक जानकारी ध्यानसँ सुनू आ समय पर जांच कराउ।",
		"हम स्पष्ट आ एके गति सँ बाजि रहल छी।",
	],
	magahi: [
		"प्रणाम, हमर नाव [अपन नाव] हई आ आज हम अपन आवाज रिकॉर्ड कर रहल हिअइ।",
		"मातृ स्वास्थ्य के जानकारी ध्यान से सुनीं आ समय पर जांच कराईं।",
		"हम साफ आवाज में बराबर रफ्तार से बोल रहल हिअइ।",
	],
	english: [
		"Hello, my name is [your name], and this is my voice sample recording.",
		"Please listen to maternal health guidance carefully and take timely checkups.",
		"I am speaking clearly at a steady pace with minimal background noise.",
	],
	gujarati: [
		"નમસ્તે, મારું નામ [તમારું નામ] છે અને હું આજે મારો અવાજ રેકોર્ડ કરું છું.",
		"માતૃત્વ આરોગ્યની માહિતી ધ્યાનથી સાંભળો અને સમયસર તપાસ કરાવો.",
		"હું સ્પષ્ટ અવાજમાં અને સમાન ગતિએ બોલું છું.",
	],
};

export default function HomePage() {
	const apiBase =
		process.env.NEXT_PUBLIC_API_BASE || "https://harshil748-voiceapi.hf.space";
	const [mode, setMode] = useState("clone");
	const [language, setLanguage] = useState("hindi");
	const [gender, setGender] = useState("female");
	const [style, setStyle] = useState("default");
	const [text, setText] = useState("नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ?");
	const [speed, setSpeed] = useState(1);
	const [pitch, setPitch] = useState(1);
	const [energy, setEnergy] = useState(1);
	const [speakerFile, setSpeakerFile] = useState(null);
	const [showScriptGuide, setShowScriptGuide] = useState(false);
	const [audioUrl, setAudioUrl] = useState("");
	const [isLoading, setIsLoading] = useState(false);
	const [message, setMessage] = useState("Ready");
	const [isError, setIsError] = useState(false);

	const selectedVoice = useMemo(() => {
		const byLang = LANG_VOICE_MAP[language] || LANG_VOICE_MAP.hindi;
		return byLang[gender] || byLang.female;
	}, [language, gender]);

	const scriptLines = useMemo(() => {
		return CLONE_SCRIPT[language] || CLONE_SCRIPT.english;
	}, [language]);

	const onGenerate = async () => {
		setIsLoading(true);
		setIsError(false);
		setMessage("Generating audio...");

		try {
			let response;

			if (mode === "clone") {
				if (!speakerFile) {
					throw new Error("Please upload a speaker WAV sample for clone mode.");
				}

				const query = new URLSearchParams({
					text,
					lang: language,
					speed: String(speed),
					pitch: String(pitch),
					energy: String(energy),
					style,
				});

				const form = new FormData();
				form.append("speaker_wav", speakerFile);

				response = await fetch(`${apiBase}/clone?${query.toString()}`, {
					method: "POST",
					body: form,
				});
			} else {
				response = await fetch(`${apiBase}/synthesize`, {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						text,
						voice: selectedVoice,
						style,
						speed,
						pitch,
						energy,
						normalize: true,
					}),
				});
			}

			if (!response.ok) {
				const errorText = await response.text();
				throw new Error(errorText || `Request failed: ${response.status}`);
			}

			const blob = await response.blob();
			const url = URL.createObjectURL(blob);
			setAudioUrl(url);
			setMessage("Audio generated successfully.");
		} catch (err) {
			setIsError(true);
			setMessage(err.message || "Failed to generate audio.");
		} finally {
			setIsLoading(false);
		}
	};

	return (
		<main className='main'>
			<div className='container'>
				<header className='header'>
					<h1 className='title'>VoiceAPI Studio</h1>
				</header>

				<section className='card' style={{ marginBottom: 16 }}>
					<div className='row'>
						<button
							className={`toggle ${mode === "clone" ? "active" : ""}`}
							onClick={() => setMode("clone")}>
							Custom Voice Clone
						</button>
						<button
							className={`toggle ${mode === "standard" ? "active" : ""}`}
							onClick={() => setMode("standard")}>
							Standard Voice
						</button>
					</div>
				</section>

				<section className='grid grid-2'>
					<div className='card'>
						<label className='label'>Language</label>
						<select
							className='select'
							value={language}
							onChange={(e) => setLanguage(e.target.value)}>
							{LANGUAGES.map((lang) => (
								<option key={lang} value={lang}>
									{lang}
								</option>
							))}
						</select>

						<label className='label' style={{ marginTop: 14 }}>
							Voice Gender
						</label>
						<div className='row'>
							<button
								className={`toggle ${gender === "male" ? "active" : ""}`}
								onClick={() => setGender("male")}
								type='button'>
								Male
							</button>
							<button
								className={`toggle ${gender === "female" ? "active" : ""}`}
								onClick={() => setGender("female")}
								type='button'>
								Female
							</button>
						</div>
						<p className='message'>Selected API voice: {selectedVoice}</p>

						<label className='label' style={{ marginTop: 14 }}>
							Style
						</label>
						<select
							className='select'
							value={style}
							onChange={(e) => setStyle(e.target.value)}>
							{STYLE_PRESETS.map((preset) => (
								<option key={preset} value={preset}>
									{preset}
								</option>
							))}
						</select>

						<div className='sliderWrap'>
							<div className='sliderLabel'>
								<span>Speed</span>
								<span>{speed.toFixed(2)}</span>
							</div>
							<input
								className='slider'
								type='range'
								min='0.5'
								max='2'
								step='0.05'
								value={speed}
								onChange={(e) => setSpeed(Number(e.target.value))}
							/>
						</div>
						<div className='sliderWrap'>
							<div className='sliderLabel'>
								<span>Pitch</span>
								<span>{pitch.toFixed(2)}</span>
							</div>
							<input
								className='slider'
								type='range'
								min='0.5'
								max='2'
								step='0.05'
								value={pitch}
								onChange={(e) => setPitch(Number(e.target.value))}
							/>
						</div>
						<div className='sliderWrap'>
							<div className='sliderLabel'>
								<span>Energy</span>
								<span>{energy.toFixed(2)}</span>
							</div>
							<input
								className='slider'
								type='range'
								min='0.5'
								max='2'
								step='0.05'
								value={energy}
								onChange={(e) => setEnergy(Number(e.target.value))}
							/>
						</div>
					</div>

					<div className='card'>
						<label className='label'>Text</label>
						<textarea
							className='textarea'
							value={text}
							onChange={(e) => setText(e.target.value)}
							placeholder='Type text to synthesize'
						/>

						<label className='label' style={{ marginTop: 14 }}>
							Speaker WAV Sample (required in clone mode)
						</label>
						<input
							className='file'
							type='file'
							accept='audio/wav'
							onChange={(e) => setSpeakerFile(e.target.files?.[0] || null)}
						/>

						{mode === "clone" && (
							<>
								<button
									type='button'
									className='guideButton'
									onClick={() => setShowScriptGuide((prev) => !prev)}>
									{showScriptGuide ?
										"Hide What To Speak"
									:	"What should I speak in the sample WAV?"}
								</button>

								{showScriptGuide && (
									<div className='guidePanel'>
										<p className='guideTitle'>
											Recommended script ({language})
										</p>
										<ol className='guideList'>
											{scriptLines.map((line, idx) => (
												<li key={`${language}-${idx}`}>{line}</li>
											))}
										</ol>
										<p className='guideTips'>
											Tips: Record 8-15 seconds, quiet room, hold phone 15-20 cm
											away, avoid music/noise, speak naturally.
										</p>
									</div>
								)}
							</>
						)}

						<button
							className='button'
							onClick={onGenerate}
							disabled={isLoading}
							style={{ marginTop: 14 }}>
							{isLoading ? "Generating..." : "Generate Audio"}
						</button>

						<p className={`message ${isError ? "error" : ""}`}>{message}</p>

						{audioUrl && (
							<div className='audioPanel'>
								<audio controls src={audioUrl} style={{ width: "100%" }} />
								<a
									className='button'
									href={audioUrl}
									download='voiceapi_output.wav'
									style={{
										display: "inline-block",
										marginTop: 10,
										textAlign: "center",
										textDecoration: "none",
									}}>
									Download WAV
								</a>
							</div>
						)}
					</div>
				</section>
			</div>
		</main>
	);
}
