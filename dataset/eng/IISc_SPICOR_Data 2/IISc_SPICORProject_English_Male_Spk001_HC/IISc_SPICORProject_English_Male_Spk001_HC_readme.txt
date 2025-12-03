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
Catalogue Id: SPICOR_ENGLISH_M_HC
Data Size (HH:MM:SS): 47 hours:52 mins:18 secs
Data Size (# Sentences): 25159
Data size (# Speakers): Male(1)
Speaker Tag: Spk0001
Domains: AGRI, ENTE, EVAL, FOOD, HEAL, LIBR, LJSP, OTHE, POLI, WEAT
Annotation file availability: YES
Recording Specifications: 48 kHz, 24 bits per sample, Mono channel
Validation status: Validated
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

Total Audio Duration:    47 hours:52 mins:18 secs
Average Sample Duration: 6.85 secs
Total Sentences:         25159
Unique word Count:       50273
Distribution of domains:
+----------+--------------------------+---------+----------------------+
| Domain   | Duration                 |   Count | Domain Description   |
+==========+==========================+=========+======================+
| AGRI     | 5 hours:4 mins:25 secs   |    2442 | AGRICULTURE          |
+----------+--------------------------+---------+----------------------+
| ENTE     | 10 hours:10 mins:12 secs |    4879 | ENTERTAINMENT        |
+----------+--------------------------+---------+----------------------+
| EVAL     | 0 hours:31 mins:22 secs  |    400  | EVALUATION           |
+----------+--------------------------+---------+----------------------+
| FOOD     | 1 hours:17 mins:3 secs   |    633  | FOOD                 |
+----------+--------------------------+---------+----------------------+
| HEAL     | 4 hours:56 mins:55 secs  |    2394 | HEALTH               |
+----------+--------------------------+---------+----------------------+
| LIBR     | 3 hours:45 mins:3 secs   |    2306 | LIBRI                |
+----------+--------------------------+---------+----------------------+
| LJSP     | 9 hours:1 mins:6 secs    |    4989 | LJSPEECH             |
+----------+--------------------------+---------+----------------------+
| OTHE     | 3 hours:21 mins:30 secs  |    2301 | OTHERS               |
+----------+--------------------------+---------+----------------------+
| POLI     | 4 hours:34 mins:26 secs  |    2218 | POLITICS             |
+----------+--------------------------+---------+----------------------+
| WEAT     | 5 hours:10 mins:12 secs  |    2596 | WEATHER              |
+----------+--------------------------+---------+----------------------+

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
