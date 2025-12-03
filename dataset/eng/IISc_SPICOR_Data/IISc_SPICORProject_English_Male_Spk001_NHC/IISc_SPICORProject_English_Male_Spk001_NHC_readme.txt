## SPICOR TTS CORPUS

As a part of the SPICOR project, studio-recorded English Male TTS data is being released.
Validated audio and text files are made available to the public. This will potentially open up
opportunities for academic researchers, students, small and large-scale industries and research
labs to innovate and develop algorithms and text-to-speech synthesizers in all the Indian languages
included in the SPICOR project.

---

## English Male Data Attributes

Type: Speech and Text
Language(s): English
Linguality: Monolingual
Catalogue Id: SPICOR_ENGLISH_M_NHC
Data Size (HH:MM:SS): 0 hours:8 mins:3 secs
Data Size (# Sentences): 77
Data size (# Speakers): Male(1)
Speaker Tag: Spk0001
Domains: AGRI, ENTE, HEAL, LIBR, LJSP, OTHE, POLI, WEAT
Annotation file availability: YES
Recording Specifications: 48 kHz, 24 bits per sample, Mono channel
Validation status: Not Validated
Data creator: Indian Institute of Sciences (IISc), Bengaluru
Year of publishing: 2024
Suggested research purpose/ areas: TTS

---

## File Structure

The corpus is organized into the following directory structure:
[Corpus_Root]/
└── [Speaker_1]/
      ├──[wavs]
      │    ├── [IISc_SPICORProject_<languageTag><genderTag><domainTag><uniqueID>.wav]
      │    ├── [IISc_SPICORProject<languageTag><genderTag><domainTag><uniqueID>.wav]
      │    ├── [IISc_SPICORProject<languageTag><genderTag><domainTag><uniqueID>.wav]
      │    ├── [...]
      │    └── [IISc_SPICORProject<languageTag><genderTag><domainTag><uniqueID>.wav]
      ├── [IISc_SPICORProject<languageTag><genderTag><speakerTag><qualityCheckTag>Transcripts.json]
      └── [IISc_SPICORProject<languageTag><genderTag><speakerTag><qualityCheckTag>_readme.txt]

---

## English Male Data Statistics

Total Audio Duration:    0 hours:8 mins:3 secs
Average Sample Duration: 6.368 secs
Total Sentences:         77
Unique word Count:       920
Distribution of domains:
+----------+------------------------+---------+----------------------+
| Domain   | Duration               |   Count | Domain Description   |
+==========+========================+=========+======================+
| AGRI     | 0 hours:0 mins:9 secs  |      1  | AGRICULTURE          |
+----------+------------------------+---------+----------------------+
| ENTE     | 0 hours:1 mins:11 secs |      10 | ENTERTAINMENT        |
+----------+------------------------+---------+----------------------+
| HEAL     | 0 hours:0 mins:27 secs |      6  | HEALTH               |
+----------+------------------------+---------+----------------------+
| LIBR     | 0 hours:1 mins:24 secs |      10 | LIBRI                |
+----------+------------------------+---------+----------------------+
| LJSP     | 0 hours:0 mins:34 secs |      5  | LJSPEECH             |
+----------+------------------------+---------+----------------------+
| OTHE     | 0 hours:3 mins:11 secs |      35 | OTHERS               |
+----------+------------------------+---------+----------------------+
| POLI     | 0 hours:0 mins:30 secs |      4  | POLITICS             |
+----------+------------------------+---------+----------------------+
| WEAT     | 0 hours:0 mins:34 secs |      5  | WEATHER              |
+----------+------------------------+---------+----------------------+

Samples in EVALUATION domain are recommended for testing TTS models. These samples belong to
various domain specific sentences, conversational sentences and erroneous sentences.

---

## Speaker MetaData

Language: English
Gender: Male
Age: 44
Experience: 6 Years
Languages known: English, Hindi, Maithilli
Mother tongue: Maithili

---

## Recording Setup

Microphone: ZOOM H6
Recording Environment: Professional studio
Recording Conditions: Studio quality at ~40dB SNR

---

## Transcription JSON Structure

Keys
├── MetaData (Project Information)
├── SpeakersMetaData (Speakers' Metadata)
└── Transcripts
        ├──[IISc_SPICORProject_<languageTag><genderTag><domainTag><uniqueID>]
        │ 			├──Transcript
        │ 			└──Domain
        │ 		
        ├──[IISc_SPICORProject<languageTag><genderTag><domainTag>_<uniqueID>]
        │ 			├──Transcript
        │ 			└──Domain
        │
        └──[...]

---

## Copyright and license

TTS data created under SPICOR project by Indian Institute of Science, Bengaluru is available
at (https://spiredatasets.iisc.ac.in/spicortts10) and the copyright in the TTS data belongs to
Indian Institute of Science, Bengaluru and the said TTS data is released or distributed under
CC-BY-4.0 license (https://creativecommons.org/licenses/by/4.0/legalcode.en). The user of
said TTS data is referred to the disclaimer of warranties section in the CC-BY-4.0 license
agreement.

---

## Acknowledgments

We extend our heartfelt gratitude to the talented voice artist whose contributions were
fundamental to this project's success.
We acknowledge the facility of IISc Studio in creating the TTS corpus.

---

## Citation

@misc{SPICOR TTS_1.0 Corpus,
     	Title = {SPICOR TTS_1.0 Corpus - A 97+ hour domain-rich Indian English TTS Corpus},
     	Authors = {Abhayjeet Et al.},
     	Year = {2025}
}

---

## Contact Information

SPIRE Lab, EE Dept., IISc, Bengaluru
Email: spirelab.ee@iisc.ac.in>

---
