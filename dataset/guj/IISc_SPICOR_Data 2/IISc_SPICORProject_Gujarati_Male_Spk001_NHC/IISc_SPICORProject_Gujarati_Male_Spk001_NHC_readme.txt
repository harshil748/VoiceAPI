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
Catalogue Id: SPICOR_GUJARATI_M_NHC
Data Size (HH:MM:SS): 4 hours:48 mins:20 secs
Data Size (# Sentences): 1366
Data size (# Speakers): Male(1)
Speaker Tag: Spk0001
Domains: AGRI, CONV, ENTE, FEST, FINA, GENE, GMIN, HEAL, HIST, INCL, MICR, POLI, RELI, SCFI, SCIE, SPAC, 
SPOR, STBK, STOR
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

## Gujarati Male Data Statistics

Total Audio Duration:    4 hours:48 mins:20 secs
Average Sample Duration: 12.674 secs
Total Sentences:         1366
Unique word Count:       14026
Distribution of domains:
+----------+-------------------------+---------+-------------------------+
| Domain   | Duration                |   Count | Domain Description      |
+==========+=========================+=========+=========================+
| AGRI     | 0 hours:2 mins:48 secs  |     14  | AGRICULTURE             |
+----------+-------------------------+---------+-------------------------+
| CONV     | 0 hours:0 mins:59 secs  |     11  | CONVERSATIONAL          |
+----------+-------------------------+---------+-------------------------+
| ENTE     | 0 hours:17 mins:26 secs |     82  | ENTERTAINMENT           |
+----------+-------------------------+---------+-------------------------+
| FEST     | 0 hours:0 mins:46 secs  |     5   | FESTIVAL                |
+----------+-------------------------+---------+-------------------------+
| FINA     | 0 hours:37 mins:48 secs |     177 | FINANCE                 |
+----------+-------------------------+---------+-------------------------+
| GENE     | 0 hours:14 mins:55 secs |     71  | GENERAL                 |
+----------+-------------------------+---------+-------------------------+
| GMIN     | 0 hours:0 mins:27 secs  |     5   | GRAMMATICALLY INCORRECT |
+----------+-------------------------+---------+-------------------------+
| HEAL     | 0 hours:34 mins:39 secs |     161 | HEALTH                  |
+----------+-------------------------+---------+-------------------------+
| HIST     | 0 hours:12 mins:27 secs |     58  | HISTORY                 |
+----------+-------------------------+---------+-------------------------+
| INCL     | 0 hours:0 mins:31 secs  |     4   | INDIAN CULTURE          |
+----------+-------------------------+---------+-------------------------+
| MICR     | 0 hours:3 mins:57 secs  |     22  | MICROSOFT               |
+----------+-------------------------+---------+-------------------------+
| POLI     | 0 hours:9 mins:3 secs   |     40  | POLITICS                |
+----------+-------------------------+---------+-------------------------+
| RELI     | 0 hours:8 mins:7 secs   |     38  | RELIGION                |
+----------+-------------------------+---------+-------------------------+
| SCFI     | 0 hours:1 mins:4 secs   |     6   | SCIENCE AND FICTION     |
+----------+-------------------------+---------+-------------------------+
| SCIE     | 2 hours:0 mins:24 secs  |     553 | SCIENCE AND TECHNOLOGY  |
+----------+-------------------------+---------+-------------------------+
| SPAC     | 0 hours:0 mins:55 secs  |     4   | SPACE                   |
+----------+-------------------------+---------+-------------------------+
| SPOR     | 0 hours:18 mins:42 secs |     98  | SPORTS                  |
+----------+-------------------------+---------+-------------------------+
| STBK     | 0 hours:2 mins:28 secs  |     12  | STORY BOOK              |
+----------+-------------------------+---------+-------------------------+
| STOR     | 0 hours:0 mins:45 secs  |     4   | STORIES                 |
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
