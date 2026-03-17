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

const PERFECT_CLONE_SCRIPT = [
	"Hello, my name is [your name], and this is my clean voice sample for cloning.",
	"Today I am speaking naturally, with steady pace, clear pronunciation, and consistent volume.",
	"Please include numbers one to ten, short questions, and normal sentence endings for better prosody.",
	"For example: How are you today? I am fine. The date is twenty-first March, and the time is nine thirty.",
];

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
	const [mousePos, setMousePos] = useState({ x: 420, y: 180 });

	const selectedVoice = useMemo(() => {
		const byLang = LANG_VOICE_MAP[language] || LANG_VOICE_MAP.hindi;
		return byLang[gender] || byLang.female;
	}, [language, gender]);

	const scriptLines = PERFECT_CLONE_SCRIPT;

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

	const onMouseMove = (event) => {
		setMousePos({ x: event.clientX, y: event.clientY });
	};

	return (
		<main
			className='main'
			onMouseMove={onMouseMove}
			style={{
				"--mx": `${mousePos.x}px`,
				"--my": `${mousePos.y}px`,
			}}>
			<div className='sparkLayer' aria-hidden='true' />
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
											Perfect clone script (speak this exactly)
										</p>
										<ol className='guideList'>
											{scriptLines.map((line, idx) => (
												<li key={`${language}-${idx}`}>{line}</li>
											))}
										</ol>
										<p className='guideTips'>
											Tips: Record 10-20 seconds in one take, quiet room, 15-20
											cm from mic, no music/reverb, no clipping, and use the
											same language as your synthesis text for best quality.
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
