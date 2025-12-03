## SPICOR TTS CORPUS

As a part of the SPICOR project, studio-recorded Gujarati Male TTS data is being released.
Validated audio and text files are made available to the public. This will potentially open up
opportunities for academic researchers, students, small and large-scale industries and research
labs to innovate and develop algorithms and text-to-speech synthesizers in all the Indian languages
included in the SPICOR project.

---

## Gujarati Male Data Attributes

Type: Speech and Text
Language(s): Gujarati
Linguality: Monolingual
Catalogue Id: SPICOR_GUJARATI_M_HC
Data Size (HH:MM:SS): 23 hours:38 mins:15 secs
Data Size (# Sentences): 7835
Data size (# Speakers): Male(1)
Speaker Tag: Spk0001
Domains: AGRI, CONV, ENTE, FEST, FINA, GENE, GMIN, HEAL, HIST, INCL, MICR, POLI, RELI, SCFI, SCIE, SPAC, 
SPOR, STBK, STOR
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

## Gujarati Male Data Statistics

Total Audio Duration:    23 hours:38 mins:15 secs
Average Sample Duration: 10.862 secs
Total Sentences:         7835
Unique word Count:       42534
Distribution of domains:
+----------+-------------------------+---------+-------------------------+
| Domain   | Duration                |   Count | Domain Description      |
+==========+=========================+=========+=========================+
| AGRI     | 0 hours:21 mins:49 secs |    122  | AGRICULTURE             |
+----------+-------------------------+---------+-------------------------+
| CONV     | 0 hours:15 mins:33 secs |    189  | CONVERSATIONAL          |
+----------+-------------------------+---------+-------------------------+
| ENTE     | 1 hours:45 mins:35 secs |    569  | ENTERTAINMENT           |
+----------+-------------------------+---------+-------------------------+
| FEST     | 0 hours:2 mins:25 secs  |    15   | FESTIVAL                |
+----------+-------------------------+---------+-------------------------+
| FINA     | 2 hours:43 mins:49 secs |    871  | FINANCE                 |
+----------+-------------------------+---------+-------------------------+
| GENE     | 1 hours:42 mins:43 secs |    563  | GENERAL                 |
+----------+-------------------------+---------+-------------------------+
| GMIN     | 0 hours:7 mins:45 secs  |    95   | GRAMMATICALLY INCORRECT |
+----------+-------------------------+---------+-------------------------+
| HEAL     | 2 hours:30 mins:25 secs |    829  | HEALTH                  |
+----------+-------------------------+---------+-------------------------+
| HIST     | 0 hours:56 mins:23 secs |    309  | HISTORY                 |
+----------+-------------------------+---------+-------------------------+
| INCL     | 0 hours:2 mins:34 secs  |    16   | INDIAN CULTURE          |
+----------+-------------------------+---------+-------------------------+
| MICR     | 0 hours:38 mins:38 secs |    246  | MICROSOFT               |
+----------+-------------------------+---------+-------------------------+
| POLI     | 0 hours:45 mins:11 secs |    238  | POLITICS                |
+----------+-------------------------+---------+-------------------------+
| RELI     | 0 hours:45 mins:5 secs  |    253  | RELIGION                |
+----------+-------------------------+---------+-------------------------+
| SCFI     | 0 hours:2 mins:20 secs  |    14   | SCIENCE AND FICTION     |
+----------+-------------------------+---------+-------------------------+
| SCIE     | 9 hours:14 mins:49 secs |    2923 | SCIENCE AND TECHNOLOGY  |
+----------+-------------------------+---------+-------------------------+
| SPAC     | 0 hours:2 mins:56 secs  |    16   | SPACE                   |
+----------+-------------------------+---------+-------------------------+
| SPOR     | 1 hours:33 mins:20 secs |    531  | SPORTS                  |
+----------+-------------------------+---------+-------------------------+
| STBK     | 0 hours:1 mins:40 secs  |    8    | STORY BOOK              |
+----------+-------------------------+---------+-------------------------+
| STOR     | 0 hours:5 mins:4 secs   |    27   | STORIES                 |
+----------+-------------------------+---------+-------------------------+

Samples in EVALUATION domain are recommended for testing TTS models. These samples belong to
various domain specific sentences, conversational sentences and erroneous sentences.

---

## Speaker MetaData

Language: Gujarati
Gender: Male
Age: 23
Experience: 1 Year
Languages known: English, Hindi, Gujarati
Mother tongue: Gujarati

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
at (https://spiredatasets.iisc.ac.in/spicortts20) and the copyright in the TTS data belongs to
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

@misc{SPICOR TTS_2.0 Corpus,
     	Title = {SPICOR TTS_2.0 Corpus - A 57+ hour domain-rich Gujarati TTS Corpus},
     	Authors = {Abhayjeet Et al.},
     	Year = {2025}
}

---

## Contact Information

SPIRE Lab, EE Dept., IISc, Bengaluru
Email: spirelab.ee@iisc.ac.in>

---
