# High-entropy correct answers: manual review

The 10 questions whose temperature-0 answer the judge graded CORRECT but whose 10 samples have the highest semantic entropy, from `pipeline_results.jsonl` (198 graded questions). Each is labelled by inspection with the main cause of the high entropy: a **clustering split** (same-meaning answers put in different clusters), a **wrong merge** (different answers put in one cluster), or **real disagreement** (the samples give genuinely different answers).

Main cause: clustering split 6, real disagreement 4. No example has a wrong merge as its main cause; wrong merges appear as a secondary issue in 2 (qz_4151, qb_3466).

## 1. In La Cage Aux Folles, what was La Cage Aux Folles?

- **id:** `tc_1968`
- **Gold answer:** Nightclub
- **Primary answer (graded CORRECT):** La Cage Aux Folles is a French nightclub in the musical, famous for its flamboyant drag performances and as the setting where the story’s gay couple runs the club.
- **Semantic entropy:** 2.303 (10 clusters)
- **Classification:** clustering split. Most samples say it is a gay nightclub but add different details (city, owners), so strict two-way entailment splits them. Four samples describe the film or musical, but still name the nightclub; some disagreement on details is real.

| Cluster | Sampled answer |
|---|---|
| 0 | La Cage aux Folles is a French comedy film (1970), a play (1965), and a musical – the story centers on a flamboyant gay nightclub called “La Cage aux Folles” in Nice, France. |
| 1 | It was a 1978 French musical comedy film (and later a Broadway show) about a gay couple running a nightclub, “La Cage aux Folles.” |
| 2 | La Cage Aux Folles is the flamboyant gay nightclub in St. Tropez that the couple owns and runs. |
| 3 | A gay nightclub in Paris. |
| 4 | La Cage Aux Folles is a French‑language musical (and later film) about a gay couple who own and run a cabaret in Nice. |
| 5 | A flamboyant gay nightclub (the “cage” for the performers) in the musical/film “La Cage aux Folles.” |
| 6 | A flamboyant gay nightclub in Paris that serves as the main setting of the musical. |
| 7 | La Cage Aux Folles was a 1975 French comedy film (later a Broadway musical and Hollywood remake) about a gay couple who run a flamboyant nightclub in St. Tropez. |
| 8 | A flamboyant gay nightclub in Paris, owned by a drag performer and his husband. |
| 9 | La Cage Aux Folles was a famous gay‑themed cabaret (drag club) in Monte Carlo, the setting for the musical and film. |

## 2. What was the name of the ferry that capsized after leaving Zeebrugge in 1987

- **id:** `qz_4151`
- **Gold answer:** The Herald of free Enterprise
- **Primary answer (graded CORRECT):** Herald of Free Enterprise
- **Semantic entropy:** 2.164 (9 clusters)
- **Classification:** real disagreement. All ten samples are wrong (none says Herald of Free Enterprise), spread over 9 clusters; also one wrong merge ("M/S Zeebrugge" with "MV ZEEBRAU").

| Cluster | Sampled answer |
|---|---|
| 0 | M/S Zeebrugge. |
| 0 | The ferry was the **MV ZEEBRAU**. |
| 1 | Princess Victoria |
| 2 | MV Albatros |
| 3 | Ferry of the Netherlands |
| 4 | The ferry was the **MV St. John**. |
| 5 | MS Prins van Oranje |
| 6 | The ferry was the MV *Fennia*. |
| 7 | MS Braemar |
| 8 | Nieuw Amsterdam |

## 3. In the US, where would a busboy work?

- **id:** `qb_5210`
- **Gold answer:** Restaurant
- **Primary answer (graded CORRECT):** In the U.S., a busboy works in the dining area of a restaurant (or hotel restaurant, banquet hall, or other food‑service venue).
- **Semantic entropy:** 2.164 (9 clusters)
- **Classification:** clustering split. All ten samples say a restaurant; extra detail in each answer splits them into 9 clusters.

| Cluster | Sampled answer |
|---|---|
| 0 | A busboy works in the service staff of a restaurant, often in the dining area, clearing tables, refilling water, and assisting servers. |
| 1 | In a restaurant (e.g., a hotel restaurant, diner, or any dining establishment). |
| 2 | In a U.S. restaurant—typically a sit‑down dining establishment, a steakhouse, café, or buffet. |
| 3 | In a U.S. restaurant or diner (typically in the dining‑room service staff). |
| 4 | In a restaurant (or other food‑service establishment such as a bar or café). |
| 5 | In a U.S. restaurant’s dining room or kitchen area—clearing tables, refilling water, and assisting servers. |
| 6 | At a restaurant (usually in the dining area). |
| 6 | In a restaurant or dining establishment. |
| 7 | In a restaurant (the dining room/kitchen area of a restaurant). |
| 8 | A busboy works in a restaurant, typically in the dining area and kitchen, clearing tables and assisting chefs. |

## 4. What kind of an organism is a truffle?

- **id:** `tc_2736`
- **Gold answer:** Fungus
- **Primary answer (graded CORRECT):** A truffle is a type of fungus (a basidiomycete).
- **Semantic entropy:** 1.973 (8 clusters)
- **Classification:** clustering split. All ten samples say a fungus; extra taxonomic detail splits them into 8 clusters.

| Cluster | Sampled answer |
|---|---|
| 0 | A truffle is a type of underground fungus (a mycorrhizal fungal fruiting body). |
| 1 | A truffle is a fungus, specifically a type of subterranean mushroom from the family Tuber. |
| 2 | A truffle is a type of edible fungus (mushroom). |
| 3 | A truffle is a type of fungus (an ectomycorrhizal basidiomycete). |
| 4 | A truffle is a type of fungus (a subterranean fruiting body of certain basidiomycete species). |
| 5 | A truffle is a type of fungus (basidiomycete). |
| 5 | A truffle is a type of fungus (specifically a basidiomycete). |
| 5 | A truffle is a type of fungus (basidiomycete). |
| 6 | A truffle is a subterranean mushroom, specifically a fruiting body of a fungus. |
| 7 | A fungus. |

## 5. What type of seaweed grows up to 100 meters in length, and is farmed for its alginates?

- **id:** `tc_2818`
- **Gold answer:** Kelp
- **Primary answer (graded CORRECT):** Kelp (Laminaria species).
- **Semantic entropy:** 1.973 (8 clusters)
- **Classification:** clustering split. Nine of ten samples say kelp, split by the species named; one says Sargassum.

| Cluster | Sampled answer |
|---|---|
| 0 | Kelp (e.g., the Laminariales such as *Ecklonia* or *Laminaria*). |
| 1 | Kelp (large brown seaweed of the order Laminariales). |
| 2 | Giant kelp (Macrocystis pyrifera). |
| 2 | Macrocystis pyrifera (giant kelp) |
| 2 | Macrocystis pyrifera – giant kelp. |
| 3 | Sargassum. |
| 4 | Laminaria (kelp). |
| 5 | Macrocystis (giant kelp). |
| 6 | Laminaria hyperborea (giant kelp). |
| 7 | Laminaria hyperborea (the kelp). |

## 6. Which Scottish physician and novelist created Dr Finlay?

- **id:** `qb_7244`
- **Gold answer:** A J Cronin
- **Primary answer (graded CORRECT):** A. J. Cronin.
- **Semantic entropy:** 1.973 (8 clusters)
- **Classification:** real disagreement. Mostly wrong names (Andrew Carnegie, several Sir Williams); only one sample says A. J. Cronin.

| Cluster | Sampled answer |
|---|---|
| 0 | Sir William McGowan. |
| 1 | Andrew Carnegie. |
| 1 | Andrew Carnegie |
| 1 | Andrew Carnegie. |
| 2 | Sir William Watson‑Gilbert. |
| 3 | Sir William Murray (Sir William M. Murray). |
| 4 | Sir William Fraser. |
| 5 | A. J. Cronin. |
| 6 | Sir William (William) — the Scottish physician and novelist Sir William — created Dr Finlay. |
| 7 | Sir William Douglas Ross. |

## 7. In the human body, what is stored in adipose tissue?

- **id:** `qb_3003`
- **Gold answer:** Fat
- **Primary answer (graded CORRECT):** Triglycerides (fat) and associated energy stores, along with hormones and signaling molecules.
- **Semantic entropy:** 1.973 (8 clusters)
- **Classification:** clustering split. All ten samples say fat or triglycerides; extra detail splits them into 8 clusters.

| Cluster | Sampled answer |
|---|---|
| 0 | Triglyceride fat (energy reserve), along with some hormones and adipocyte cells. |
| 1 | Energy in the form of triglycerides (fat). |
| 1 | Energy as triglycerides (fat). |
| 1 | Triglyceride fat (stored energy). |
| 2 | Lipids (fat) are stored in adipose tissue. |
| 3 | Triglycerides (fat), along with small amounts of water, proteins, and hormone‑producing substances. |
| 4 | Triglycerides (fat). |
| 5 | Triglycerides (fat) – stored as lipid droplets in adipose cells. |
| 6 | Triglyceride fats (and some cholesterol), which serve as stored energy and also produce adipokine hormones. |
| 7 | Triglyceride (fat) molecules (plus some stored hormones and insulation). |

## 8. what is the 21st letter of the Greek alphabet?

- **id:** `qb_3466`
- **Gold answer:** Phi
- **Primary answer (graded CORRECT):** Phi (Φ)
- **Semantic entropy:** 1.887 (7 clusters)
- **Classification:** real disagreement. Samples name Phi, Psi, Mu, Upsilon and Tau; also a wrong merge ("Φ" with "Tau") and Phi split across 3 clusters.

| Cluster | Sampled answer |
|---|---|
| 0 | Phi (ϕ). |
| 1 | Psi (Ψ) |
| 1 | Ψ (psi) |
| 2 | Mu (μ) |
| 3 | Φ |
| 3 | Tau |
| 4 | Phi. |
| 5 | Upsilon |
| 5 | Upsilon. |
| 6 | Mu (Μ, μ) |

## 9. The White Rose Shopping Centre is in which English city?

- **id:** `qb_9397`
- **Gold answer:** Leeds
- **Primary answer (graded CORRECT):** Leeds.
- **Semantic entropy:** 1.887 (7 clusters)
- **Classification:** real disagreement. Samples name Leeds, Hull, Wakefield, Scarborough and Southport; Leeds is also split across 2 clusters.

| Cluster | Sampled answer |
|---|---|
| 0 | Kingston upon Hull (Hull) |
| 0 | Kingston upon Hull (Hull). |
| 1 | Leeds. |
| 1 | Leeds, England. |
| 2 | Leeds, West Yorkshire. |
| 2 | Leeds, West Yorkshire. |
| 3 | Scarborough, North Yorkshire, England. |
| 4 | Wakefield. |
| 5 | Southport, Merseyside. |
| 6 | Wakefield, West Yorkshire. |

## 10. Which teeth are the third molars, and are always the last to0 erupt?

- **id:** `tc_2900`
- **Gold answer:** Wisdom teeth
- **Primary answer (graded CORRECT):** The third molars are the wisdom teeth, located at the very back of the mouth (four of them, two on each side). They are always the last teeth to erupt.
- **Semantic entropy:** 1.834 (7 clusters)
- **Classification:** clustering split. All ten samples say wisdom teeth; extra (sometimes wrong) detail splits them into 7 clusters.

| Cluster | Sampled answer |
|---|---|
| 0 | The third molars are the wisdom teeth, the last teeth to erupt. |
| 0 | The third molars are the wisdom teeth—the last set of molars that normally erupt. |
| 0 | Third molars (wisdom teeth) are the last teeth to erupt. |
| 1 | The third molars (wisdom teeth) are the lower back teeth numbered 48 and 49 and the upper back teeth numbered 38 and 39, and they are always the last teeth to erupt. |
| 2 | Third molars (wisdom teeth) are the four teeth at the back of the mouth—two upper and two lower—and they are the last to erupt. |
| 2 | Third molars are the wisdom teeth, one in each quadrant of the upper and lower jaws. They are the last teeth to erupt. |
| 3 | The third molars are the wisdom teeth, the fourth set of molars at the back of the mouth. |
| 4 | The third molars are the wisdom teeth (the lower second molars and the upper second molars), which are the last teeth to erupt. |
| 5 | The third molars are the wisdom teeth, also called the fourth molars, and they are the last teeth to erupt. |
| 6 | The third molars are the wisdom teeth – the four teeth in the corners of the mouth (the lower right, lower left, upper right and upper left) commonly numbered 48, 38, 33, 32. They are the last teeth to erupt. |
