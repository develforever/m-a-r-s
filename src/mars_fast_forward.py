"""
mars_fast_forward.py — wektoryzowany forward dla M.A.R.S. (Etap B / B+).

Rozwiązuje problem z faza2_mnist.json: throughput 0.59× (M.A.R.S. wolniejszy
od baseline mimo mniejszej liczby MAC), spowodowany pętlą `for pod_id` z
maskowaniem (na GPU = N osobnych kernel launchy).

Trzy strategie (wszystkie poprawne, różna wydajność wg sprzętu i N):
  - forward_grouped (V2): sortuj próbki po podzie, jeden matmul na grupę.
    Najlepszy dla MAŁEGO N (mało podów).
  - forward_loopless (V3): bmm z paddingiem, 1 kernel dla wszystkich podów.
    Najlepszy dla DUŻEGO N na GPU (1 kernel zamiast N).
  - forward_auto: automatycznie wybiera V2/V3 wg progu N. Zawsze najlepszy.

Wagi podów trzymane jako stacked tensory [N_pods, in, out] — to klucz
do wektoryzacji (zamiast ModuleList z osobnymi Linear, jak w mars_torch.py).

ZMIERZONE (GTX 1050 Ti, sesja 2026-06-15):
  Najlepszy reżim: hidden=2048, N=8, V2 grouped → 2.57× szybszy od monolitu.
  Crossover V2/V3: ~N=16 (poniżej V2 wygrywa, powyżej V3).
  Słodki punkt rozmiaru poda: hidden≈2048 (przy 4096 baseline sam sycí GPU).
"""
import torch
import torch.nn as nn


class FastPods(nn.Module):
    """
    N podów o identycznym kształcie, trzymanych jako stacked tensory.
    Każdy pod: Linear(n_in, hidden) -> ReLU -> Linear(hidden, n_out).
    """

    # próg przełączania strategii (zmierzony na GTX 1050 Ti)
    # N <= próg: V2 (grouped) wygrywa; N > próg: V3 (loopless, 1 kernel) wygrywa
    AUTO_SWITCH_N = 16

    def __init__(self, n_pods, n_in, hidden, n_out):
        super().__init__()
        self.n_pods, self.n_in, self.hidden, self.n_out = n_pods, n_in, hidden, n_out
        self.W1 = nn.Parameter(torch.randn(n_pods, n_in, hidden) / (n_in ** 0.5))
        self.b1 = nn.Parameter(torch.zeros(n_pods, hidden))
        self.W2 = nn.Parameter(torch.randn(n_pods, hidden, n_out) / (hidden ** 0.5))
        self.b2 = nn.Parameter(torch.zeros(n_pods, n_out))

    def forward_grouped(self, x, ids):
        """V2: sortuj po podzie, matmul per grupa. Najlepszy dla małego N."""
        out = torch.zeros(x.shape[0], self.n_out, device=x.device, dtype=x.dtype)
        order = torch.argsort(ids)
        x_s = x[order]
        counts = torch.bincount(ids, minlength=self.n_pods)
        start = 0
        for pid in range(self.n_pods):
            c = int(counts[pid].item())
            if c > 0:
                h = torch.relu(x_s[start:start+c] @ self.W1[pid] + self.b1[pid])
                out[order[start:start+c]] = h @ self.W2[pid] + self.b2[pid]
                start += c
        return out

    def forward_loopless(self, x, ids):
        """V3: bmm z paddingiem, 1 kernel dla wszystkich podów. Najlepszy dla dużego N na GPU."""
        B = x.shape[0]
        order = torch.argsort(ids)
        x_s, ids_s = x[order], ids[order]
        counts = torch.bincount(ids, minlength=self.n_pods)
        max_c = int(counts.max().item())
        padded = torch.zeros(self.n_pods, max_c, self.n_in, device=x.device, dtype=x.dtype)
        pos_in_group = (torch.cat([torch.arange(int(c), device=x.device)
                                   for c in counts.tolist()])
                        if B > 0 else torch.zeros(0, dtype=torch.long, device=x.device))
        padded[ids_s, pos_in_group] = x_s
        h = torch.relu(torch.bmm(padded, self.W1) + self.b1.unsqueeze(1))
        o = torch.bmm(h, self.W2) + self.b2.unsqueeze(1)
        out = torch.zeros(B, self.n_out, device=x.device, dtype=x.dtype)
        out[order] = o[ids_s, pos_in_group]
        return out

    def forward_loop(self, x, ids):
        """V0: stara pętla z maską — TYLKO do porównania/poprawności."""
        out = torch.zeros(x.shape[0], self.n_out, device=x.device, dtype=x.dtype)
        for pid in range(self.n_pods):
            m = ids == pid
            if m.any():
                h = torch.relu(x[m] @ self.W1[pid] + self.b1[pid])
                out[m] = h @ self.W2[pid] + self.b2[pid]
        return out

    def forward_auto(self, x, ids):
        """
        Automatyczny wybór najlepszej strategii wg liczby podów.
        Z pomiarów GPU: V2 (grouped) najlepszy dla małego N, V3 (loopless)
        dla dużego N (crossover ~N=16). Daje ZAWSZE najlepszy z dwóch
        wariantów — darmowe wyciśnięcie wydajności (jedna linijka if).
        """
        if self.n_pods <= self.AUTO_SWITCH_N:
            return self.forward_grouped(x, ids)
        return self.forward_loopless(x, ids)

    def forward(self, x, ids):
        """Domyślne forward = auto-wybór strategii."""
        return self.forward_auto(x, ids)


if __name__ == "__main__":
    # test poprawności: wszystkie warianty dają to samo
    torch.manual_seed(0)
    pods = FastPods(n_pods=10, n_in=784, hidden=64, n_out=10)
    x = torch.randn(1000, 784)
    ids = torch.randint(0, 10, (1000,))
    with torch.no_grad():
        o0 = pods.forward_loop(x, ids)
        o2 = pods.forward_grouped(x, ids)
        o3 = pods.forward_loopless(x, ids)
        oa = pods.forward_auto(x, ids)
    checks = {
        "grouped == loop": torch.allclose(o0, o2, atol=1e-4),
        "loopless == loop": torch.allclose(o0, o3, atol=1e-4),
        "auto == loop": torch.allclose(o0, oa, atol=1e-4),
    }
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print("Wszystkie warianty poprawne." if all(checks.values())
          else "BŁĄD: warianty się różnią!")
