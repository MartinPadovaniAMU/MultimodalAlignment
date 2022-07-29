# [EN]

This repository contains inputs, source files and outputs for three levels of multimodal alignment on adult/adult and child/adult conversations based on the ChiCo Corpus.

## Mimicry

**Input :** parts of the [BC repository](https://github.com/afourtassi/BC), and more precisely the `data_cog` directory, from the ChiCo corpus. Please fill the `input.txt` file with the path of your cloned local repository.

**Sources :** `MIMICRY.ipynb` and [the mimicry function](https://github.com/kelhad00/IBPY/blob/master/interaction_analysis.py)

**Output :** PNG plots at `output/mimicry/`

## Lexical Alignment

**Output :** `output/lexical alignment.csv`

## Speech Rate Alignment

**Input :** conversation files (.csv), annotations of the different game phases from the ChiCo corpus, word.csv and the version 3.83 of the [Lexique database](http://www.lexique.org/) by New & Pallier.

**Sources :** `SPEECH_RATE.ipynb`, `MY_FUNCTIONS.ipynb`

**Output :** PNG plots at `output/speech_rate/`
