import "./globals.css";

export const metadata = {
	title: "VoiceAPI Studio",
	description:
		"Custom voice cloning and multilingual TTS frontend for VoiceAPI",
};

export default function RootLayout({ children }) {
	return (
		<html lang='en'>
			<body>{children}</body>
		</html>
	);
}
