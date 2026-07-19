# Droga L — plan pre-rejestrowany: fork tożsamości (mocniejszy zamrożony backbone)

Data pre-rejestracji: 2026-07-17 (branch `droga-l`; main/v0.6 nietknięty).
Kontekst: `PLAN_GENERALNY.md` Etap L; przesłanka pomiarowa: K0 — sufit
losowych cech CIFAR = 39.65, mechanizm realizuje 94.6% sufitu, luka
30.6pp do joint jest w całości reprezentacyjna.

## Deklaracja forka (jawna, do narracji)

To jest ŚWIADOMA zmiana osi zasobów: dotąd CAŁY backbone był losowy
(zero pretrainingu — RELATED_WORK pozycjonuje projekt tą osią);
tu podmieniamy warstwę (1) na publiczny pretrenowany encoder. Stare
liczby zostają nietknięte i pozostają główną linią „from-scratch";
seria L raportowana jest osobno jako linia „foundation-embedding".
Mechanizm (2)(3) i protokół kolektywny (Droga I) pozostają BEZ ZMIAN —
L jest testem twierdzenia „representation-agnostic" z WHITEPAPER.

## Encoder (decyzja pre-rejestrowana)

ResNet18-ImageNet (torchvision, `IMAGENET1K_V1`) — publiczny,
deterministyczny, dostępny bez dodatkowych pakietów; pretraining:
1.28M obrazów ImageNet (oś zasobów raportowana jawnie). Zamrożony,
zawsze w eval (BatchNorm z pretrenowanymi statystykami).

Interfejs zachowany: cechy 512-d (avgpool) → ZAMROŻONA LOSOWA projekcja
512→128 (seed agenta) → ReLU. Dzięki temu D=128 jak dotąd: te same
pody, ta sama pamięć klasy (24.1 KB przy k16), ten sam payload
protokołu I. Wejście: denormalizacja CIFAR → normalizacja ImageNet →
resize 224 (bilinear). Wspólny seed nadal synchronizuje agentów
(pretrained część identyczna z definicji, projekcja z seeda).

## L1 — single-agent Split-CIFAR-10 na mocnych cechach

Pliki: `src/mars_cl_l.py` (PretrainedBackbone),
`src/run_L1_pretrained.py` → `results/L1_pretrained.json`

Warianty (CIFAR-n, kotwice 50d — K0/K1: kotwice nie są gardłem CIFAR):
  l1_all  : sufit zamrożonych cech pretrained (proj_train="all",
            analog K0)
  l1_seq  : uczciwy CL sekwencyjny, sparse_k16 (konfiguracja J2b,
            zmieniony tylko backbone)

**Kryteria (Z GÓRY, class-IL):**
- Główne: l1_seq vs J2b sparse_k16 (37.51 ± 1.35, TE SAME seedy, pary):
  SYGNAL+/−/parowy±/SZUM. Oczekiwany SYGNAL+, ale nietrywialny:
  cechy ImageNet na CIFAR-32 (resize ×7) mogą być słabe — werdykt
  symetryczny.
- Diagnostyka (analog K0): gap_mech_L = l1_all − l1_seq (czy mechanizm
  nadal realizuje >90% sufitu na lepszej reprezentacji); dystans
  l1_seq do joint 70.24 (czy fork zamyka lukę reprezentacyjną).
- Ryzyko pre-rejestrowane: losowa projekcja 512→128 może zdusić
  cechy pretrained (informacja w 512-d, my bierzemy losowy rzut);
  jeśli l1_all ≈ 39.65 (sufit losowych), fork nie działa przez rzut,
  nie przez encoder — rozróżnienie raportowane.

## L2 — kolektyw na mocnych cechach (protokół I bez zmian)

Plik: `src/run_L2_collective_cifar.py` → `results/L2_collective_cifar.json`

Setup = run_I3 przeniesiony na CIFAR-n z PretrainedBackbone:
5 agentów × 2 klasy, kolektor = agent 0, 4 adopcje, n_dream=5000
(parytet klasy CIFAR), payload identyczny (24.1 KB).

**Kryteria (Z GÓRY):**
- Główne: kolektyw vs l1_seq (pary per-seed, te same seedy):
  SZUM = równoważność potwierdza przenośność protokołu na nową
  reprezentację i trudniejsze dane; SYGNAL− = limit protokołu.
- Obserwacje: krzywa po adopcjach; luka do l1_all.

## Kolejność uruchomień u Roberta

1. `python src/mars_cl_l.py` (smoke jednostkowy; pierwszy run pobierze
   wagi resnet18 przez torchvision — wymaga internetu)
2. `python src/run_L1_pretrained.py --smoke`, potem FULL
   (ekstrakcja cech przez resnet18@224 — rząd kilku minut na 1050 Ti)
3. `python src/run_L2_collective_cifar.py --smoke`, potem FULL
   (wymaga results/L1_pretrained.json jako bazy par)

Wyniki do DROGA_L_NOTATKI.md; merge po komplecie i decyzji Roberta.

## Dopisek implementacyjny (2026-07-19, przed FULL — bez zmiany semantyki)

Pierwsze podejście liczyło cechy resnet18@224 w każdym przebiegu
(feats_batched + eval po każdym tasku) — na GTX 1050 Ti nieakceptowalnie
wolno. Ponieważ część pretrained jest deterministyczna i WSPÓLNA dla
wszystkich seedów/agentów, jej wyjście [N, 512] liczone jest RAZ
i cache'owane (`data/cifar_resnet18_224_feats.pt`,
`extract_or_load_cifar_feats`); per seed pozostaje losowa projekcja
512→128 + ReLU (`ReducedBackbone`). Złożenie identyczne
z `PretrainedBackbone` — kryteria i warianty planu bez zmian.
