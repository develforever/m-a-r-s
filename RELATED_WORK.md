# Related Work — zweryfikowana bibliografia i szkic sekcji

Status: 2026-07-10; aktualizacja 2026-07-23 — Części F (unlearning, seria N)
i G (poisoning/kolektyw niezaufany, seria I4), 11 nowych pozycji
zweryfikowanych. KAŻDA pozycja poniżej zweryfikowana web searchem (tytuł,
autorzy, wenue, link). Zero cytowań z pamięci. Szkic sekcji po angielsku —
do wklejenia do WHITEPAPER.md (lub do papera CL po rozcięciu).

---

## Część A — Zweryfikowana bibliografia

### A1. Continual learning — problem, protokoły, baseline'y

1. **[EWC]** Kirkpatrick et al., *Overcoming catastrophic forgetting in neural
   networks*, PNAS 114(13):3521–3526, 2017.
   https://www.pnas.org/doi/10.1073/pnas.1611835114
   — nasz baseline; nasza obserwacja "EWC nie działa w class-IL" jest zgodna
   z literaturą (patrz [Three-scenarios] poniżej).
2. **[Three-scenarios]** van de Ven & Tolias, *Three scenarios for continual
   learning*, arXiv:1904.07734, 2019 (NeurIPS CL workshop).
   https://arxiv.org/abs/1904.07734
   — kanoniczna definicja task-/domain-/class-IL; pokazuje, że metody
   regularyzacyjne (EWC) zawodzą w class-IL. Cytować przy definicji protokołu
   i przy wyniku F0.
3. **[iCaRL]** Rebuffi, Kolesnikov, Sperl, Lampert, *iCaRL: Incremental
   Classifier and Representation Learning*, CVPR 2017, arXiv:1611.07725.
   https://arxiv.org/abs/1611.07725
   — klasyk exemplar-based class-IL; NCM na przechowywanych egzemplarzach.
   My: NCM/prototypy BEZ egzemplarzy.
4. **[GDumb]** Prabhu, Torr, Dokania, *GDumb: A Simple Approach that Questions
   Our Progress in Continual Learning*, ECCV 2020 (oral).
   https://link.springer.com/chapter/10.1007/978-3-030-58536-5_31
   — krytyka pola: trywialny greedy-buffer bije dedykowane metody CL.
   Uzasadnia nasz wybór replay jako JEDYNEGO poważnego przeciwnika.
5. **[Robins95]** Robins, *Catastrophic Forgetting, Rehearsal and
   Pseudorehearsal*, Connection Science 7(2):123–146, 1995.
   https://www.ingentaconnect.com/content/tandf/ccos/1995/00000007/00000002/art00002
   — protoplasta "snu": rehearsal bez przechowywania danych. Nasz sen
   parametryczny to pseudorehearsal ze statystyk, nie z szumu.

### A2. Zamrożone reprezentacje + prototypy/statystyki (najbliżsi sąsiedzi)

6. **[NCM]** Mensink, Verbeek, Perronnin, Csurka, *Distance-Based Image
   Classification: Generalizing to New Classes at Near-Zero Cost*, TPAMI 2013.
   https://inria.hal.science/hal-00817211/en
   — fundament klasyfikacji prototypowej; nowe klasy przez średnią cech.
7. **[SLDA]** Hayes & Kanan, *Lifelong Machine Learning with Deep Streaming
   Linear Discriminant Analysis*, CVPRW 2020, arXiv:1909.01520.
   https://arxiv.org/abs/1909.01520
   — zamrożony ekstraktor + strumieniowa LDA (wspólna kowariancja).
8. **[FeTrIL]** Petit, Popescu, Schindler, Picard, Delezoide, *FeTrIL: Feature
   Translation for Exemplar-Free Class-Incremental Learning*, WACV 2023,
   arXiv:2211.13131. https://arxiv.org/abs/2211.13131
   — zamrożony ekstraktor + pseudo-cechy starych klas przez translację
   geometryczną centroidów. NAJBLIŻSZY nam mechanizm pseudo-cech.
9. **[FeCAM]** Goswami, Liu, Twardowski, van de Weijer, *FeCAM: Exploiting the
   Heterogeneity of Class Distributions in Exemplar-Free Continual Learning*,
   NeurIPS 2023, arXiv:2309.14062. https://arxiv.org/abs/2309.14062
   — prototypy + kowariancje klas (Mahalanobis) na zamrożonym backbone.
   UWAGA: nasz negatyw H1b (pełna kowariancja < lokalne centroidy na cechach
   po ReLU) jest z tym w napięciu — u nich kowariancja pomaga na cechach
   ViT/ResNet; różnica: shrinkage + Tukey + silne cechy. Omówić wprost.
10. **[RanPAC]** McDonnell et al., *RanPAC: Random Projections and Pre-trained
    Models for Continual Learning*, NeurIPS 2023.
    https://papers.nips.cc/paper_files/paper/2023/hash/2793dc35e14003dd367684d93d236847-Abstract-Conference.html
    — LOSOWA zamrożona projekcja + prototypy w CL; najbliżsi nam w "losowość
    jako uczciwy komponent CL". Różnica: u nich losowa projekcja NAD
    pretrenowanym ViT; u nas losowy jest CAŁY backbone (bez pretrainingu),
    a routing kotwiczy semantyka słów.
11. **[PASS]** Zhu, Zhang, Wang, Yin, Liu, *Prototype Augmentation and
    Self-Supervision for Incremental Learning*, CVPR 2021.
    https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.html
    — rehearsal prototypów z augmentacją w przestrzeni cech, bez egzemplarzy.

### A3. Feature/generative replay i ochrona wag

12. **[GFR]** Liu, Wu, et al., *Generative Feature Replay for
    Class-Incremental Learning*, CVPRW 2020.
    https://openaccess.thecvf.com/content_CVPRW_2020/html/w15/Liu_Generative_Feature_Replay_for_Class-Incremental_Learning_CVPRW_2020_paper.html
    — replay CECH z uczonego generatora zamiast obrazów. My: bez generatora,
    czyste statystyki k-centroidowe (~KB/klasę), backbone losowy zamrożony.
13. **[BIR]** van de Ven, Siegelmann, Tolias, *Brain-inspired replay for
    continual learning with artificial neural networks*, Nature
    Communications 11:4069, 2020.
    https://www.nature.com/articles/s41467-020-17866-2
    — replay reprezentacji wewnętrznych ("sen" mózgopodobny); tylko generative
    replay ratuje class-IL. Nasz sen = ta sama intuicja, zredukowana do
    nieparametrycznych statystyk możliwych dzięki STACJONARNOŚCI zamrożonych cech.
14. **[OWM]** Zeng, Chen, Cui, Yu, *Continual learning of context-dependent
    processing in neural networks*, Nature Machine Intelligence 1:364–372, 2019.
    https://www.nature.com/articles/s42256-019-0080-x
    — ortogonalna modyfikacja wag (RLS); użyta w naszym H1 jako eliminacja
    (dryf vs wierność snu).
15. **[GFR+OWM]** Shen, Zhang, Chen, Deng, *Generative Feature Replay with
    Orthogonal Weight Modification for Continual Learning*, IEEE (IJCNN) 2021,
    arXiv:2005.03490. https://arxiv.org/abs/2005.03490
    — ISTNIEJĄCA kombinacja OWM + feature replay (jak nasze H1). Różnice:
    uczony generator vs nasze statystyki; trenowany backbone vs losowy
    zamrożony; brak kotwic semantycznych. KONIECZNIE cytować przy H1 —
    recenzent to znajdzie.

### A4. Semantyka języka jako kotwice klas

16. **[DeViSE]** Frome, Corrado, Shlens, Bengio, Dean, Ranzato, Mikolov,
    *DeViSE: A Deep Visual-Semantic Embedding Model*, NIPS 2013.
    https://www.cs.toronto.edu/~ranzato/publications/frome_nips2013.pdf
    — mapowanie cech wizualnych na embeddingi słów (zero-shot). Protoplasta
    naszych kotwic; my przenosimy to do sekwencyjnego class-IL i pokazujemy,
    że wymaga ochrony projekcji (sen), inaczej katastrofa (g1_seq).
17. **[DAP]** Lampert, Nickisch, Harmeling, *Attribute-Based Classification
    for Zero-Shot Visual Object Categorization*, TPAMI 2014 (CVPR 2009).
    https://www.semanticscholar.org/paper/3fd90098551bf88c7509521adf1c0ba9b5dfeb57
    — przepis BCE-per-atrybut użyty w G2; nasz negatyw G2 + reguła
    osiągalności strukturalnej.
18. **[GloVe]** Pennington, Socher, Manning, *GloVe: Global Vectors for Word
    Representation*, EMNLP 2014. https://aclanthology.org/D14-1162/
    — źródło naszych prototypów semantycznych.
19. **[Continual-CLIP]** Thengane et al., *CLIP model is an Efficient
    Continual Learner*, arXiv:2210.03114, 2022.
    https://arxiv.org/abs/2210.03114
    — enkoder tekstu jako klasyfikator w CL bez treningu. Różnica osiowa:
    CLIP konsumuje masywny multimodalny pretraining; my pokazujemy, że
    wystarczy STATYCZNA geometria słów (GloVe) + losowe cechy + sen.
    (Uwaga: autorów "Thengane et al." potwierdza arXiv — sprawdzić przy
    finalnym bibtexie na stronie arXiv.)
20. **[SDC]** Yu, Twardowski, Liu, Herranz, Wang, Cheng, Jui, van de Weijer,
    *Semantic Drift Compensation for Class-Incremental Learning*, CVPR 2020,
    arXiv:2004.00440. https://arxiv.org/abs/2004.00440
    — "semantic drift" = dryf embeddingów w CL, kompensowany bez egzemplarzy.
    Cytować też dla higieny terminologicznej ("semantic" u nich ≠ u nas).

### A5. Modularność i routing (Part II)

21. **[MoE-1991]** Jacobs, Jordan, Nowlan, Hinton, *Adaptive Mixtures of Local
    Experts*, Neural Computation 3(1):79–87, 1991.
    https://doi.org/10.1162/neco.1991.3.1.79
22. **[MoE-2017]** Shazeer et al., *Outrageously Large Neural Networks: The
    Sparsely-Gated Mixture-of-Experts Layer*, ICLR 2017, arXiv:1701.06538.
    https://arxiv.org/abs/1701.06538
23. **[ExpertGate]** Aljundi, Chakravarty, Tuytelaars, *Expert Gate: Lifelong
    Learning with a Network of Experts*, CVPR 2017.
    https://openaccess.thecvf.com/content_cvpr_2017/html/Aljundi_Expert_Gate_Lifelong_CVPR_2017_paper.html
    — routing do ekspertów per zadanie przez autoenkodery; pokrewne naszemu
    routingowi prototypowemu i tezie "routing = wybór zadania w class-IL".
24. **[ECOC]** Dietterich & Bakiri, *Solving Multiclass Learning Problems via
    Error-Correcting Output Codes*, JAIR 2:263–282, 1995, arXiv:cs/9501101.
    https://arxiv.org/abs/cs/9501101
    — zasada projektowa dla słownika atrybutów G2b (dystans Hamminga).

---

## Część B — Szkic sekcji "Related Work" (EN, do papera CL)

> **Related Work.**
> **Class-incremental learning and its baselines.** We follow the three-scenario
> taxonomy of van de Ven & Tolias [Three-scenarios], reporting class-IL as the
> primary protocol. Regularization methods such as EWC [EWC] are known to fail
> in class-IL, which our baselines reproduce. Rehearsal from a stored buffer
> remains the strongest simple baseline: GDumb [GDumb] showed that a greedy
> buffer embarrasses much of the specialized CL literature, which motivates our
> choice of tuned experience replay as the sole serious opponent, with
> pre-registered buffer size. iCaRL [iCaRL] combines exemplars with
> nearest-class-mean classification [NCM]; we keep the NCM decision rule but
> remove the exemplars.
> **Frozen representations with prototypes or statistics.** A growing
> exemplar-free line freezes the feature extractor and represents past classes
> by statistics: streaming LDA [SLDA], geometric translation of centroids
> [FeTrIL], prototype augmentation [PASS], and Bayes classifiers with per-class
> covariances [FeCAM]. RanPAC [RanPAC] is closest in spirit to our use of
> randomness, injecting a frozen random projection — but above a large
> pretrained ViT. All of these presuppose a strong pretrained backbone; our
> system uses a *random* frozen backbone, isolating the contribution of the CL
> mechanism itself from representation quality, and measuring the price of that
> choice explicitly (the random-feature ceiling). Our finding that full
> covariance underperforms local diagonal centroids on sparse post-ReLU
> features complements FeCAM's covariance-shrinkage recipe on pretrained
> features: density models of features are architecture-sensitive.
> **Replay without stored data.** Pseudorehearsal dates to Robins [Robins95];
> modern instantiations replay learned generative models of features [GFR] or
> internal representations [BIR], and have been combined with orthogonal
> weight protection [OWM, GFR+OWM]. Our parametric sleep differs in that the
> frozen backbone makes per-class feature distributions *stationary*, so the
> past can be stored as tiny nonparametric statistics (k-centroids, ~KB/class)
> instead of a trained generator — and our OWM experiment serves as an
> *elimination*, showing the residual gap is dream fidelity, not drift.
> **Language as class geometry.** DeViSE [DeViSE] mapped images into word-vector
> space for zero-shot recognition; attribute-based zero-shot classification
> [DAP] and CLIP-based continual learners [Continual-CLIP] likewise exploit
> textual class semantics. We transplant this idea into sequential class-IL
> with the *cheapest possible* language resource — static GloVe vectors
> [GloVe] — and measure both where it works (clothing) and where it fails
> (digit names), plus the failure of naive sequential anchoring without sleep.
> **Modular routing.** Part II relates to mixtures of experts [MoE-1991,
> MoE-2017] and task-expert selection [ExpertGate]; our contribution there is
> the pre-registered demonstration that routing is representation-limited, and
> that oracle-based headroom estimates are inflated.

---

## Część C — Pozycjonowanie: najbliżsi sąsiedzi i czym się różnimy

| Praca | Wspólne | Różnica kluczowa |
|---|---|---|
| RanPAC (NeurIPS'23) | losowość + prototypy, bez rehearsal | oni: losowa projekcja NAD pretrenowanym ViT; my: losowy CAŁY backbone + semantyczne kotwice + sen |
| FeTrIL (WACV'23) | zamrożony ekstraktor, pseudo-cechy z centroidów | oni: translacja geometryczna, ekstraktor pretrenowany na task0; my: k-centroidowe próbkowanie do treningu projekcji pod kotwice słów |
| FeCAM (NeurIPS'23) | statystyki klas na zamrożonych cechach | oni: kowariancja+Mahalanobis pomaga (ViT); nasz H1b: pełna kowariancja < lokalne centroidy (ReLU, losowe cechy) — omówić jako komplementarne |
| GFR+OWM (IJCNN'21) | OWM + feature replay | oni: uczony generator, trenowany backbone; my: statystyki bez generatora, losowy backbone, OWM jako eliminacja diagnostyczna |
| BIR (Nat.Comm.'20) | "sen" reprezentacji wewnętrznych | oni: uczony model generatywny, context gating; my: stacjonarność cech → czyste statystyki |
| Continual-CLIP (2022) | tekstowe kotwice klas w CL | oni: masywny pretraining CLIP; my: statyczne GloVe + losowe cechy — oś zasobów jawnie tańsza |

**Ryzyka recenzenckie (nazwane):**
1. "Czemu nie porównujecie z RanPAC/FeCAM na waszych benchmarkach?" —
   odpowiedź: te metody wymagają pretrenowanego backbone'u, co jest dokładnie
   zasobem, który kontrolujemy; porównanie = future work przy decyzji
   o piętrze reprezentacji (po publikacji, zgodnie z planem).
2. "Split-Fashion/MNIST to małe benchmarki" — odpowiedź: F4 (CIFAR) + rama
   "mechanizm izolowany od jakości reprezentacji"; uczciwie w limitations.
3. "GFR+OWM już połączyło te klocki" — odpowiedź przygotowana w tabeli.
4. Sprawdzić przy submisji: dokładna lista autorów RanPAC (McDonnell et al.)
   i Continual-CLIP (Thengane et al.) bezpośrednio na arXiv.

## Część F — Machine unlearning (seria N; zweryfikowane 2026-07-23)

### F1. Bibliografia

F1. **[Cao-Yang]** Cao & Yang, *Towards Making Systems Forget with Machine
   Unlearning*, IEEE S&P 2015, s. 463–480.
   https://dblp.org/rec/conf/sp/CaoY15.html
   — praca definiująca pole; unlearning przez formę sumacyjną algorytmu.
   My: nasza "forma sumacyjna" to statystyki snu — odbudowa bez danych.
F2. **[SISA]** Bourtoule et al., *Machine Unlearning*, IEEE S&P 2021,
   arXiv:1912.03817. https://arxiv.org/abs/1912.03817
   — gwarancja przez retrenowanie shardu; koszt ograniczony strukturą
   treningu. Nasz reinit (N1c) = ta sama filozofia gwarancji
   ("wytrenuj od nowa bez celu"), ale odbudowa idzie ZE SNÓW pozostałych
   klas — zero przechowanych danych, koszt sekund, nie godzin.
F3. **[Certified-removal]** Guo, Goldstein, Hannun, van der Maaten,
   *Certified Data Removal from Machine Learning Models*, ICML 2020,
   arXiv:1911.03030. https://arxiv.org/abs/1911.03030
   — definicja certyfikowanego usunięcia (nieodróżnialność od modelu,
   który danych nie widział) dla modeli liniowych. Nasza projekcja JEST
   liniowa; nasze kryterium N1c (re-learning ≈ never-seen, +0.34pp < próg
   0.50) to operacyjny, mierzony odpowiednik tej definicji.
F4. **[Scrubbing]** Golatkar, Achille, Soatto, *Eternal Sunshine of the
   Spotless Net: Selective Forgetting in Deep Networks*, CVPR 2020.
   https://openaccess.thecvf.com/content_CVPR_2020/html/Golatkar_Eternal_Sunshine_of_the_Spotless_Net_Selective_Forgetting_in_Deep_CVPR_2020_paper.html
   — przybliżone "szorowanie" wag z górnym ograniczeniem pozostałej
   informacji. Nasz wariant scrub (N1b, ~84% wymazania) to ta kategoria;
   nasza taksonomia pokazuje, że przybliżenie jest zbędne, gdy nośnik
   informacji jest znany i tani w odbudowie.
F5. **[Unlearning-survey]** Nguyen et al., *A Survey of Machine Unlearning*,
   arXiv:2209.02299. https://arxiv.org/abs/2209.02299
   — mapa pola (exact vs approximate, metryki weryfikacji). Nasza metryka
   "koszt ponownego nauczenia ze 100 obrazów" = relearn-based verification.

### F2. Pozycjonowanie serii N (do wklejenia w EN)

Trzy różnice strukturalne wobec pola: (1) **znany nośnik informacji** —
N1b dowodzi pomiarem, że projekcja jest JEDYNYM nośnikiem (pody
konfirmacyjne); literatura unlearningu zwykle traktuje sieć jako
monolit o nieznanej lokalizacji śladu. (2) **Gwarancja przez odbudowę
ze snów**: reinit + rebuild z pamięci parametrycznej pozostałych klas —
odpowiednik retreningu SISA, ale bez dostępu do jakichkolwiek danych
i przy koszcie ≤0 dla pozostałych klas (N1c). (3) **Weryfikacja
relearn-based z progiem pre-rejestrowanym** (≈ never-seen). Wynik
uboczny o wadze protokołowej: klasa nieobecna w projekcji jest
routingowo nieosiągalna nawet przy obecnej kotwicy i podzie.

## Część G — Poisoning i kolektyw niezaufany (seria I4; zweryfikowane 2026-07-23)

### G1. Bibliografia

G1. **[Biggio-SVM]** Biggio, Nelson, Laskov, *Poisoning Attacks against
   Support Vector Machines*, ICML 2012, arXiv:1206.6389.
   https://arxiv.org/abs/1206.6389 — kanoniczny atak na zbiór treningowy.
G2. **[BadNets]** Gu, Dolan-Gavitt, Garg, *BadNets: Identifying
   Vulnerabilities in the Machine Learning Model Supply Chain*, 2017,
   arXiv:1708.06733. — backdoor przez łańcuch dostaw modelu. Nasz payload
   24 KB to analogiczny "łańcuch dostaw", ale statystyk, nie wag.
G3. **[Krum]** Blanchard, El Mhamdi, Guerraoui, Stainer, *Machine Learning
   with Adversaries: Byzantine Tolerant Gradient Descent*, NeurIPS 2017.
   https://dblp.org/rec/conf/nips/BlanchardMGS17.html
   — odporna agregacja gradientów (odrzucanie odstających). U nas nie ma
   agregacji: wiadomość jest atomowa (adopcja paczki), więc obrona
   przesuwa się z agregacji na detekcję i odwołanie paczki.
G4. **[FL-backdoor]** Bagdasaryan, Veit, Hua, Estrin, Shmatikov, *How To
   Backdoor Federated Learning*, AISTATS 2020, arXiv:1807.00459.
   https://proceedings.mlr.press/v108/bagdasaryan20a.html
   — model replacement w FL; agregator ślepy na pochodzenie update'u.
   Kontrast: nasz odbiorca widzi CAŁY payload i może go w całości cofnąć.
G5. **[BrainWash]** Abbasi, Nooralinejad, Pirsiavash, Kolouri, *BrainWash:
   A Poisoning Attack to Forget in Continual Learning*, CVPR 2024,
   arXiv:2311.11995. https://arxiv.org/abs/2311.11995
   — zatruwanie danych, by wymusić zapominanie w CL. Komplementarne:
   oni atakują dane treningowe uczącego się; my payload protokołu wymiany
   — i mierzymy blast radius + pełną naprawę.
G6. **[FedEraser]** Liu, Ma, Yang, Wang, Liu, *FedEraser: Enabling
   Efficient Client-Level Data Removal from Federated Learning Models*,
   IEEE/ACM IWQoS 2021.
   — usuwanie wkładu klienta z modelu FL wymaga historii update'ów na
   serwerze. U nas naprawa nie wymaga żadnej historii: zapomnij-i-adoptuj
   -ponownie (I4b) korzysta z maszynerii serii N i payloadu źródła.

### G2. Pozycjonowanie serii I4 (do wklejenia w EN)

Model zagrożenia inny niż w FL: wymieniamy JEDNORAZOWE statystyki, nie
iterowane gradienty/wagi — nie ma pętli agregacji, którą atakuje
literatura byzantine. Trzy zmierzone twierdzenia: (1) **blast radius =
paczka adopcyjna** (klasy własne odporne ±1pp — sen ich broni; swap
niszczy obie współadoptowane klasy przez sprzeczne cele kotwic);
(2) **detekcja na losowych cechach niemożliwa** (oba pre-rejestrowane
detektory bez separacji — uczciwy negatyw; kandydat: cechy semantyczne,
seria P); (3) **pełna odwracalność bez historii** — unlearn paczki +
ponowna adopcja wraca do ścieżki clean w szumie na wszystkich metrykach
(I4b), czyli odpowiednik federated unlearning bez przechowywania
czegokolwiek poza samym payloadem. Polityka: adopcje paczkami, zasięg
naprawy = zasięg szkody.

## Część D — Braki do domknięcia przed submisją

- [ ] Finalny BibTeX każdej pozycji ze strony arXiv/DOI (nie z tego pliku).
- [ ] GDumb: dopisać arXiv ID (użyto linku Springer; arXiv istnieje).
- [ ] Rozważyć 1–2 cytowania przeglądowe CL (np. survey Wang et al. 2023,
    arXiv:2302.00487 — pojawił się w wynikach, ZWERYFIKOWAĆ przed użyciem).
- [ ] Part II (routing ceiling): jeśli osobny paper — dodać literaturę
    o task inference w class-IL i o ocenie routerów MoE (osobna runda searchy).
- [x] 2026-07-23: rundy dla serii N (unlearning, Część F) i I4
    (poisoning/kolektyw, Część G) — wykonane, 11 pozycji zweryfikowanych.

## Część E — Dopisek po serii I/L (2026-07-19): model collapse a wymiana snów

Zarzut, którego należy się spodziewać przy Serii I: „trening na danych
generowanych = model collapse" (Shumailov et al. 2023/2024, *The Curse
of Recursion* / Nature 2024; Alemohammad et al. 2023, *Self-Consuming
Generative Models Go MAD* — ZWERYFIKOWAĆ ID przy submisji). Odpowiedź
z konstrukcji protokołu, nie z nadziei:

1. **Rekursja ma głębokość 1.** Payload to statystyki policzone
   jednorazowo na cechach REALNYCH danych (zamrożony backbone); odbiorca
   zapisuje je dosłownie i nigdy nie re-estymuje ich ze snów. Sny są
   materiałem treningowym projekcji/podów, ale nie źródłem kolejnych
   statystyk — pętla sprzężenia, która napędza collapse (generacja →
   estymacja → generacja), jest przerwana strukturalnie.
2. **Miniaturowa demonstracja kosztu rekursji jest zmierzona:** jedyny
   wariant wykonujący JEDEN cykl re-estymacji ze snów (I2 `fusion_red`:
   sen z payloadu → ponowny k-means) płaci kierunkowo −0.67pp (5/5 par
   ujemnych). Jeden cykl = mierzalna strata; protokół domyślny nie
   wykonuje żadnego.
3. **Kumulacja adopcji ≠ kumulacja rekursji:** I3/L2 (4 kolejne adopcje)
   nie wykazały narastającego dryfu — każda wiadomość jest pierwszej
   generacji względem realnych danych.
4. Kontrast z federated/decentralized learning (Petals, DiLoCo,
   Bittensor — wymiana aktywacji/gradientów/wag, wąskie gardło sieci,
   problem zatrutych gradientów): tu wiadomość jest jednorazowa,
   ~24 KB/klasę, asynchroniczna i weryfikowalna semantycznie (kandydat
   I4: odbiorca śni z payloadu i sprawdza zgodność z kotwicą
   deklarowanej klasy — obrona bez kryptoekonomii).
   [AKTUALIZACJA 2026-07-23: I4 wykonane — detekcja na losowym backbone
   NEGATYWNA (oba detektory), naprawa przez unlearn+readopt PEŁNA;
   detekcja semantyczna = kandydat serii P. Literatura: Część G.]
