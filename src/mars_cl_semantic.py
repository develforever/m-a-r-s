"""
mars_cl_semantic.py -- G1: prototypy semantyczne (slowa jako kotwice).

IDEA (pomysl uzytkownika, DROGA_G_PLAN.md):
  Prototypem klasy NIE jest srednia obrazow, lecz WEKTOR SLOWA (GloVe 50d,
  zamrozony). Prototyp istnieje ZANIM system zobaczy pierwszy obraz klasy:
  nowa klasa = nowe slowo = prototyp za darmo. Uczymy tylko projekcje
  obraz -> przestrzen slow. Backbone: losowy zamrozony (zwyciezca F1d/F2).

Dlaczego to moze przelamac plateau F2 (~65%):
  Losowe cechy + NCM osiagnely sufit; slowa wnosza informacje SPOZA obrazow
  -- strukture znaczen (koszula blizej plaszcza niz sneakersa). To pierwszy
  kanal w projekcie, ktory nie jest funkcja pikseli.

Routing: kosinusowy (embedding i prototypy normalizowane; cdist na sferze
jednostkowej = ranking kosinusowy). Trening projekcji: CE po podobienstwach
kosinusowych do slow klas (temperatura TEMP), tylko slowa klas dostepnych
w danym trybie.

Tryby projekcji (pre-rejestrowane):
  "task0": projekcja uczona TYLKO na zadaniu 0, potem zamrozona
           (czysty CL; ryzyko waskiej supervizji jak F1a -- mierzone).
  "seq"  : projekcja douczana na kazdym zadaniu (tylko nowe dane, stare
           slowa jako negatywy w CE) -- dotyka wspolnej projekcji, wiec
           ryzyko dryfu starych klas; dryf MIERZONY przez forgetting.
  "all"  : projekcja na wszystkich klasach -- DIAGNOSTYKA (gorna granica,
           nie jest uczciwym CL).

BONUS mierzalny: ZERO-SHOT ROUTING -- po nauce projekcji na zadaniu 0
prototypy WSZYSTKICH 10 klas juz istnieja (slowa!); mierzymy routing na
klasach, ktorych projekcja nigdy nie widziala.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mars_cl import MarsCLSystem

TEMP = 0.1
GLOVE_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "data",
                                  "glove.6B.50d.txt")

# Slowa per klasa (usredniane; dobrane z zapasem na braki w slowniku GloVe)
CLASS_WORDS = {
    "Fashion-MNIST": {
        0: ["t-shirt", "tee"], 1: ["trousers", "pants"],
        2: ["pullover", "sweater"], 3: ["dress", "gown"],
        4: ["coat", "jacket"], 5: ["sandal", "sandals"],
        6: ["shirt", "blouse"], 7: ["sneaker", "sneakers"],
        8: ["bag", "handbag"], 9: ["boot", "boots"],
    },
    "MNIST": {i: [w] for i, w in enumerate(
        ["zero", "one", "two", "three", "four",
         "five", "six", "seven", "eight", "nine"])},
    # CIFAR-10: nazwy semantycznie bogate (warunek stosowalnosci z G1/F3b)
    "CIFAR-10": {
        0: ["airplane", "plane"], 1: ["automobile", "car"],
        2: ["bird"], 3: ["cat"], 4: ["deer"], 5: ["dog"],
        6: ["frog"], 7: ["horse"], 8: ["ship"], 9: ["truck"],
    },
}


def load_word_vectors(ds_name, glove_path=GLOVE_PATH_DEFAULT, device="cpu"):
    """
    Wczytuje z GloVe TYLKO potrzebne slowa (jedno przejscie po pliku).
    Zwraca dict: klasa -> znormalizowany tensor [50].
    """
    words_per_class = CLASS_WORDS[ds_name]
    needed = {w for ws in words_per_class.values() for w in ws}
    found = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            tok, rest = line.split(" ", 1)
            if tok in needed:
                found[tok] = torch.tensor(
                    [float(v) for v in rest.split()], device=device)
                if len(found) == len(needed):
                    break
    vecs = {}
    for c, ws in words_per_class.items():
        have = [found[w] for w in ws if w in found]
        if not have:
            raise ValueError(f"Brak slow {ws} (klasa {c}) w {glove_path}")
        vecs[c] = F.normalize(torch.stack(have).mean(dim=0), dim=0)
    missing = needed - set(found)
    if missing:
        print(f"  (uwaga: slowa poza slownikiem GloVe, pominiete: {missing})")
    return vecs


class MarsCLSemantic(MarsCLSystem):
    def __init__(self, word_vecs, proj_train="task0", channels=(8, 16),
                 pod_hidden=24, backbone_module=None):
        assert proj_train in ("task0", "seq", "all")
        emb_dim = len(next(iter(word_vecs.values())))
        super().__init__(backbone_source="random", proto_mode="mean",
                         emb_dim=emb_dim, pod_hidden=pod_hidden,
                         channels=channels, backbone_module=backbone_module)
        self.word_vecs = {c: v.clone() for c, v in word_vecs.items()}
        self.proj_train = proj_train

    # embedding kosinusowy (na sferze jednostkowej)
    def embed_from_feats(self, feats):
        return F.normalize(self.proj(feats), dim=1)

    # ------------------------------------------------------------ projekcja
    def _fit_proj(self, X, y, classes, epochs, lr, device):
        """CE po podobienstwach kosinusowych do slow podanych klas."""
        W = torch.stack([self.word_vecs[c].to(device) for c in classes])
        c2i = {c: i for i, c in enumerate(classes)}
        yi = torch.tensor([c2i[int(v)] for v in y.tolist()], device=device)
        crit = nn.CrossEntropyLoss()
        for p in self.proj.parameters():
            p.requires_grad = True
        opt = torch.optim.Adam(self.proj.parameters(), lr=lr)
        for _ in range(epochs):
            perm = torch.randperm(len(X), device=device)
            for s in range(0, len(X), 512):
                idx = perm[s:s + 512]
                with torch.no_grad():
                    feats = self.backbone(X[idx])
                emb = F.normalize(self.proj(feats), dim=1)
                logits = emb @ W.T / TEMP
                loss = crit(logits, yi[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        for p in self.proj.parameters():
            p.requires_grad = False

    def init_representation(self, task_data, epochs, lr, device):
        # backbone: losowy, zamrozony (zwyciezca F1d/F2). Projekcja wg trybu.
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        if self.proj_train == "task0":
            td = task_data[0]
            self._fit_proj(td["Xtr"], td["ytr"], td["classes"],
                           epochs, lr, device)
        elif self.proj_train == "all":
            X = torch.cat([td["Xtr"] for td in task_data])
            y = torch.cat([td["ytr"] for td in task_data])
            classes = sorted(c for td in task_data for c in td["classes"])
            self._fit_proj(X, y, classes, epochs, lr, device)
        # "seq": projekcja uczona w learn_task

    # ------------------------------------------------------------ nauka
    def learn_task(self, td, epochs, lr, device):
        classes = td["classes"]
        if self.proj_train == "seq":
            # douczanie projekcji na danych zadania; stare slowa = negatywy
            self._fit_proj(td["Xtr"], td["ytr"],
                           self.seen_classes + list(classes),
                           epochs, lr, device)
        for c in classes:
            self.protos[c] = self.word_vecs[c].to(device)
        self.seen_classes = self.seen_classes + list(classes)
        self._train_pods(classes, td["Xtr"], td["ytr"], epochs, lr, device)

    # ------------------------------------------------------------ zero-shot
    @torch.no_grad()
    def zero_shot_routing(self, task_data, trained_classes, device):
        """
        Routing po WSZYSTKICH 10 slowach-prototypach (istnieja a priori),
        mierzony osobno na klasach widzianych/niewidzianych przez projekcje.
        Nie zmienia stanu modelu.
        """
        saved_protos, saved_seen = dict(self.protos), list(self.seen_classes)
        self.protos = {c: self.word_vecs[c].to(device) for c in range(10)}
        self.seen_classes = list(range(10))
        accs = {}
        for td in task_data:
            Xv, yv = td["Xte"], td["yte"]
            routed = self.route(self.embed(Xv))
            acc = (routed == yv).float().mean().item()
            kind = ("seen" if all(c in trained_classes
                                  for c in td["classes"]) else "unseen")
            accs[f"task_{td['classes']}"] = {"routing_acc": round(acc, 4),
                                             "kind": kind}
        self.protos, self.seen_classes = saved_protos, saved_seen
        return accs


if __name__ == "__main__":
    # Smoke bez GloVe: losowe "wektory slow".
    torch.manual_seed(0)
    wv = {c: F.normalize(torch.randn(50), dim=0) for c in range(10)}
    X = torch.randn(400, 784)
    y = torch.cat([torch.full((100,), c) for c in range(4)])
    td0 = {"classes": [0, 1], "Xtr": X[:200], "ytr": y[:200],
           "Xte": X[:100], "yte": y[:100]}
    td1 = {"classes": [2, 3], "Xtr": X[200:], "ytr": y[200:],
           "Xte": X[300:], "yte": y[300:]}
    m = MarsCLSemantic(wv, proj_train="task0")
    m.init_representation([td0, td1], epochs=1, lr=1e-3, device="cpu")
    m.learn_task(td0, epochs=1, lr=1e-3, device="cpu")
    m.learn_task(td1, epochs=1, lr=1e-3, device="cpu")
    assert m.forward(X[:16]).shape == (16, 10)
    zs = m.zero_shot_routing([td0, td1], trained_classes=[0, 1], device="cpu")
    assert len(zs) == 2
    print("Smoke test OK.", zs)
