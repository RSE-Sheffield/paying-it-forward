Definitions
We assign a scale for both research software and data to determine an artifact has been generated, used or not relevant by the research.

By ‘Substantive’, we mean software or data that is materially involved in producing, analysing, validating, or enabling the research results, rather than being used only for incidental, administrative, or routine purposes. In other words, it is about whether new knowledge has been created.


Definition
S1 Research Software Generated

The paper should describe how the research created or significantly extended research software. Crucially, the software created should generate new knowledge.


S2 Research Software Used
The paper should describe how the research uses existing research software. Crucially, the software used should generate new knowledge.

S3 No Research Software
Either software is not used, or the software used in the research does not create new knowledge, but instead acts as a utility.

D1 Research Substantive Data Generated
The paper should describe how the research created or significantly extended research data. Crucially, the data created should generate new knowledge.

D2 Research Substantive Data Used
The paper should describe how the research used existing research data. Crucially, the data should generate new knowledge.

D3 No Research Data
Either research data is not used, or the Substantive Data Used in the research does not create new knowledge.

Some General Rules

If a paper both uses and generates software/data, then it belongs in the generated category. In other words, we assume if software/data is generated, it is also used.

If the appropriate category is unclear, require clear evidence before classifying software or data as generated. Where the paper does not clearly establish that the software or data was produced by the research, classify it as used instead.

We take the paper as the ground truth when calculating this metric. For example, if they say they have generated software or data, but don’t provide a way of accessing it, then we still consider this S1 or D1. Future Work packages will assess the data stewardship/fairness of the research.

Artefacts must have been used or created in the research reported in the paper. Planned or proposed artefacts are not considered. For example, statements such as “We will use software X” or “We will collect data Y” should not be classified as evidence of software or data use/generation. This is particularly common in research protocols; such papers should be classified as S3D3 unless they provide evidence of actual software or data use/generation.


Research Data and Software is defined as:

Table 1: The definitions of research software and data we are using

Definition	Typical evidence in a paper
Research Data	"Research data can be defined as 'recorded factual material commonly retained by and accepted in the scientific community as necessary to validate research findings'. Although the majority of such data is created in digital format, all research data is included in the definition, irrespective of the format in which it is created." – UKRI, 2026	Dataset, corpus, database, annotations, benchmark data, supplementary data, model weights, lab books

Research Software	Research Software includes source code files, scripts, computational workflows and executables that were created during the research process or for a research purpose. Software Components (e.g. operating systems, libraries, dependencies, packages, scripts, etc.) that are used for research but were not created during or with a clear research intent should be considered software in research and not Research Software. We have taken the definition from 10.5281/zenodo.5504015 but have removed algorithms from the definition.	Source code, package, scripts, application, model implementation

Other	Research artefacts that are neither data nor software	Hardware, physical materials, protocols, instruments, figures, documentation, experimental designs, or no research artefacts produced outside the paper at all (e.g. a literature review)


