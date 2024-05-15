---
theme: default
layout: intro
lineNumbers: true
defaults:
    transition: slide-up
---

<!-- LTeX: language=it -->

# Energy games solver
Domiziano Scarcelli - 1872664 - A.A 2023/2024

<!--
Buongiorno, in questa presentazione vorrei descrivere in che maniera ho realizzato un risolutore di energy games.
-->

---
layout: section
---

# Games on Graphs
Branca della teoria dei giochi che descrive i giochi le cui regole e l'evoluzione sono rappresentate da un grafo

<!--
Innanzitutto ci troviamo nel contesto di "Games on Graphs", ovvero una branca della teoria di giochi che descrive quei giochi le cui regole e l'evoluzione sono rappresentati da un grafo.

Prima di cimentarmi nella definizione di Energy Games, vorrei parlare un po' della teoria necessaria per risolvere i problemi a questi legati.
-->

---
layout: center
---

# Arena

- Il posto in cui si svolge il gioco

<v-clicks>

- Insieme di vertici $V = V_\text{Min} \cup V_\text{Max}$
    - Finiti: arena finita
    - Infiniti: arena infinita
- Giocatore singolo o multipli. Nel nostro caso 2 giocatori: Min e Max

</v-clicks>

<!--
La componente principale di un gioco è l'Arena, cioè il posto dove si svogle il gioco. 
[CLICK]
Questa è modellata da un grafo che possiede un insieme di vertici e archi.
[CLICK]
Il gioco può essere giocato da uno o più giocatori. Nel caso degli energy games i giocatori sono due, che chiameremo Min e Max.
-->

---
layout: center
---

# Play
- Token si muove da un vertice all'altro

<v-clicks>

- Il giocatore che "possiede" il vertice $v$ muove il token attraverso un arco $v \rightarrow v'$
- La sequenza di movimenti del token denota una giocata (play) $\pi$.
    - Finita o infinita, ma non vuota

</v-clicks>

<!--
All'interno del gioco è presente un Token, il quale è posizionato su un vertice iniziale e
[CLICK]
può essere mosso solamente dal giocatore a cui appartiene quel vertice.
[CLICK]
La sequenza di movimenti del token attraverso gli archi del grafo denota una giocata.
Da notare che in un gioco una giocata può essere finita o infinita, ma mai vuota.
-->

---
layout: center
---

# Strategie
- Funzione che mappa giocate finite in archi.

<v-clicks>

- Denotata con $\sigma: \text{Paths} \rightarrow E$
- $\text{Paths} = \{\pi_0, \pi_1, \dots,. \pi_n\}$ insieme di $n$ giocate.
- Definisce una descrizione completa del comportamento di un giocatore per ogni possibile situazione.

</v-clicks>

<!--
Una strategia è definita come una funzione che mappa giocate finite in archi. 
[CLICK]
Denotata in questa maniera, 
[CLICK]
dove Paths è la sequenza delle n giocate.
[CLICK]
Questa definisce una descrizione completa del comportamento di un giocatore per ogni possibile situazione, ovvero l'arco su cui il giocatore deve muovere il token, data la precedente serie di mosse.
-->

---
layout: center
---

# Condizioni
- ***Winning condition***: condizione che definisce quando un giocatore vince.
    - **Qualitativa**: è una funzione che separa le giocate vincenti $W \subset \text{Paths}$ da quelle non vincenti.
    - **Quantitativa**: è una funzione che assegna un punteggio (valore reale o $\pm \infty$) ad ogni giocata.

<!-- Affinché il gioco possa essere concluso, una condizione vincente deve essere soddisfatta.
La condizione dipende dal tipo di gioco e può essere qualitativa o quantitativa.
La prima è una funzione che separa giocate vincenti da quelle non vincenti.
La seconda invece assegna un punteggio a ogni giocata. --> 

---
layout: center
---

# Strategie vincenti 
- In un gioco **qualitativo**, una strategia $\sigma$ è vincente da un vertice $v$ se ogni giocata che parte da $v$ consistente con $\sigma$ è vincente.

<v-click>

- In un gioco **quantitativo**, una strategia $\sigma$ è vincente da un vertice $v$ se per ogni giocata che parte da $v$ consistente con $\sigma$ ha un valore maggiore di una certa soglia $x \in \mathbb{R}$. Ovvero $f(\pi) \ge x$.

</v-click>

<!--
In un gioco qualitativo, una strategia è vincente da un vertice $v$ se ogni giocata che parte da $v$ consistente con la strategia è vincente.
[CLICK]
In un gioco quantitativo si aggiunge come condizione quella di avere un valore maggiore di una certa soglia $x$.
-->

---
layout: section
---

# Energy games
Definizione del gioco e degli obiettivi

<!-- Ora che abbiamo le conoscenze essenziali per modellare un risolutore di un gioco, possiamo scendere nel dettaglio di un particolare tipo di giochi chiamato Energy Games. -->

---
layout: center
---

# Energy Games
- Gioco quantitativo (a somma zero) a 2 giocatori (Min e Max)

<v-clicks>

- Arena: $\mathcal{A} = (G, V_\text{Min}, V_\text{Max})$
- Grafo: $G = (V, E, In, Out)$

</v-clicks>

<v-click>

Ogni arco $e \in E$ ha un peso $w(e) \in [-W, W]$ con $W \in \mathbb{R}$

</v-click>

<!--
Un energy game è un gioco a somma zero giocato da due giocatori su
[CLICK]
un arena definita nella seguente maniera. 
[CLICK]
Quindi definisce anche un grafo i cui vertici sono bipartiti tra Min e Max. 
[CLICK]
Su ogni arco è presente un peso che è definito su un intervallo [-W, W].
-->

---
layout: center
---

# Energia
- Peso dell'arco denota accumulo o consumo di energia:
    - Peso positivo: giocatore accumula energia
    - Peso negativo: giocatore consuma energia

<v-click>

## Obiettivi
- Min: trovare un valore iniziale tale che possa navigare il grafo all'infinito mantenendo un'energia non negativa.
- Max: impedire l'obiettivo di Min.

Il gioco è a somma zero: se Min vince, Max perde e viceversa.

</v-click>

<!--
Questo peso denota un accumulo di energia per il giocatore che vi passa sopra, in caso il peso è positivo, o una perdita di energia in caso il peso è negativo. 
[CLICK]
L'obiettivo del gioco è quello di trovare il valore minimo iniziale tale che il giocatore Min possa navigare il grafo all'infinito mantenendo un'energia non negativa. 

Essendo il gioco a somma zero, l'obiettivo di Max è quello di impedire l'obiettivo di Min.
-->

---
layout: center
---

# Problemi computazionali associati
Cosa possiamo calcolare

- Risolvere il gioco: booleano che indica se Min ha una strategia vincente

<v-clicks>

- Calcolare il valore del gioco: modellare funzione di valore (valore dell'energia iniziale di Min che soddisfa la condizione vincente, per ogni vertice $v \in V$.)
- Costruire una strategia ottimale per Min.

</v-clicks>

<v-click>

Il risolutore implementato calcola il valore del gioco.

</v-click>

<v-click>

>Notare che l'arena ha una soluzione diversa da $+\infty$ solo se non esistono cicli negativi.

</v-click>

<!--
I problemi computazionali associati agli energy games sono i seguenti: 

- Risolvere il gioco, il che consiste nel costruire un algoritmo che dato un gioco, ritorni un booleano che indica se Min possiede una strategia vincente o meno.
[CLICK]
- Calcolare il valore del gioco, che consiste nel modellare la funzione di valore, ovvero trovare il valore dell'energia iniziale di Min che gli permetta di vincere il gioco, questo per ogni vertice V.
[CLICK]
- Costruire una strategia ottimale per min.
[CLICK]
Il risolutore implementato nel progetto si focalizza nel calcolare il valore del gioco.
[CLICK]
È importante notare che un gioco è risolvibile per Min solo se il grafo non presenta cicli negativi, ovvero cicli la cui somma dei pesi è negativa. Questo perchè il giocatore Max potrebbe costringere Min a percorrere il ciclo infinite volte, il che richiederebbe un'energia iniziale infinita.
-->

---
layout: section
---

# Algoritmo "Value Iteration"
Trovare il valore del gioco con una complessità $O(nmW)$

<!-- Per definire il primo algoritmo che ci permetterà di risolvere il gioco dobbiamo prima definire due funzioni, che sono le seguenti. --> 

---

# Naive Value Iteration

<div class="flex flex-col h-full gap-2 items-center justify-evenly">

<div class="border-1 p-3">
$$
\delta(l, w) = \max(l-w, 0)
$$
</div>

<div class="border-1 p-3">
$$
\mathbb{O}^{\mathcal{G}}(\mu)(u)=
    \begin{cases}
        \min \{\delta(\mu(v), w): u \stackrel{w}{\rightarrow} v \in E\} & \text { if } u \in V_{\text {Min}},
        \\ \max \{\delta(\mu(v), w): u \stackrel{w}{\rightarrow} v \in E\} & \text { if } u \in V_{\text {Max}} .
    \end{cases}
$$
</div> 
</div> 


<!-- La prima calcola il delta tra due valori, limitandolo a zero in caso il valore è negativo.

La seconda invece definisce una strategia "greedy" per i giocatori Min e Max. Presi in input il nodo attuale e la funzione di valore $\mu$, ritorna il delta minimo o massimo, a seconda del giocatore, calcolato tra il valore di un nodo uscente a $u$ ed il peso dell'arco uscente, questo per ogni nodo.

Lo definisco come strategia greedy perchè per la scelta vengono considerati solamente i nodi uscenti al nodo corrente. -->

---
layout: center
---
# Pseudocodice

<div class="border-white border-1 px-6">

$\textbf{for } u \in V \textbf{do} \\$ 
$\quad \mu(u) \gets 0 \\$
$\textbf{repeat} \\$
$\quad \mu \gets \mathbb{O}^\mathcal{G}(\mu) \\$
$\textbf{until } \mu = \mathbb{O}^\mathcal{G}(u)\\$
$\textbf{return } \mu$

</div>

<!-- L'algoritmo completo è molto semplice, visto che si tratta di un algoritmo di tipo "value iteration", inizializziamo la funzione di valore a zero per ogni nodo e assegniamo a ogni nodo $u$ il risultato della funzione $O$, questo per ogni nodo. Ripetiamo questo finché i valori di tutti i nodi non cambiano tra un'iterazione all'altra.

Ora vediamo qual è la complessità computazionale di questo algoritmo. -->

---

# Analisi della complessità
- Calcolare $\mathbb{O}^\mathcal{G}(\mu)(u)$ ha una complessità di $O(|Out(u)|)$

<v-clicks>

- Sia $p$ la probabilità di avere un arco tra due vertici, il numero medio di archi uscenti da un vertice è $p \cdot n$.
- Il ciclo viene ripetuto per tutti gli $n$ nodi, quindi la complessità è $O(n^2p)$, ovvero $O(m)$.
- Il massimo numero di iterazioni prima di arrivare alla convergenza è $O(nW)$.
  
</v-clicks>

<v-clicks>

Quindi la complessità totale è $O(nmW)$.

</v-clicks>

<!--
Calcolare la funzione $O^G$ per un nodo $u$ ha una complessità pari alla somma dei nodi uscenti ad $u$.

[CLICK]
Sia $p$ la probabilità di avere un arco tra due vertici, allora il numero medio di archi uscenti da un vertice è $pn$
[CLICK]
La funzione $O^G$ viene calcolata per tutti i nodi, e quindi la complessità di una sola iterazione del "repeat" è O(n^2p), che può essere scritta come O(m) dove "m" è il numero di archi totali. Questo è vero perchè durante la costruzione del grafo noi andiamo a considerare tutte le n^2 coppie di vertici, ed inseriamo un arco tra di queste con una probabilità p, quindi m = n^2p.
[CLICK]
Visto che il massimo numero di iterazioni prima di arrivare alla convergenza è O(nW), [CLICK] la complessità totale dell'algoritmo è O(nmW).
-->

---

# Implementazione Python
<div class="no-scrollbar">

```python {1-3|4-15|16-100}{maxHeight:'500px'}
def _delta(self,l, w): 
        return max(l-w, 0)

def _O(self, node: int):
    """
    The O^G function which returns the max value between all the 
    outgoing edges from the node (if player is Max), 
    or the min value (if player is Min).
    """
    values = (self._delta(self.arena.value_mapping[v], w) for 
        (u, v, w) in self.arena.get_outgoing_edges(node))
    if self.arena.player_mapping[node] == Player.MAX:  
        return max(values, default=0)
    else:  # player is MIN
        return min(values, default=0)

def value_iteration(self):
    """
    The naive value iteration algorithm to compute the value function.
    """
    threshold = 0.000001
    steps = 0
    max_steps = 50_000
    pbar = tqdm(total=max_steps, desc="Value iteration")

    # Maximum n iterationr, so complexity is O(n^3) in the case of edge_probability = 1
    while True:
        pbar.update(1)
        steps += 1
        if steps > max_steps:
            break

        old_value = self.arena.value_mapping.copy()
        # O(n^2) complexity
        for node in self.arena.nodes:
            # O(n) complexity
            self.arena.value_mapping[node] = self._O(node)

        if all((abs(self.arena.value_mapping[node] - old_value[node]) < threshold for node in self.arena.nodes)):
            break

    pbar.close()

    if steps > max_steps:
        print(f"Naive Value Iteration - Did not converge after {steps} steps")
    else:
        print(f"Naive Value Iteration - Converged after {steps} steps")
    return steps
```
</div>

<!-- Qui possiamo vedere come l'algoritmo è stato implementato in Python. Abbiamo quindi la funzione delta e la funzione O. L'algoritmo principale quindi definisce una treshold che servirà per la convergenza dei valori. In caso il grafo presenti uno o più cicli negativi, l'algoritmo non arriverà mai a tale convergenza, ecco perchè ho inserito un massimo numero di iterazioni, dopo il quale l'algoritmo si fermerà. -->

---

# Refined Value Iteration Algorithm
Come rendere l'algoritmo più efficiente?
     
Possiamo essere più efficienti e considerare solo un sottoinsieme di nodi per ogni iterazione.

<v-clicks>

Definiamo le seguenti variabili:
 - Un insieme di vertici incorretti `Incorrecct`
 - Una mappa `Count` che associa a ogni vertice di Min il numero di archi uscenti incorretti.

</v-clicks>

<v-click>

## Definizione di "incorretto"

</v-click>

<v-clicks>

- Un arco $u \xrightarrow{w} v$ è incorretto se $\mu(u) < \delta(\mu(u), w)$ 
- Un vertice $u$ è incorretto:
  - Ha un arco uscente incorretto e $u \in V_{\text{Max}}$
  - Tutti i suoi archi uscenti sono incorretti e $u \in V_{\text{Min}}$

</v-clicks>

<!--
Nella versione più ottimizzata dell'algoritmo l'idea è quella di evitare di aggiornare il valore di tutti i nodi ad ogni iterazione, ma di considerare solo un sottoinsieme di nodi.
[CLICK][CLICK]
Chiamiamo quindi tale sottoinsieme "Incorrect". Definiamo anche un dizionario "Count" che associa ad ogni vertice di "Min" il numero di archi uscenti incorretti.
[CLICK][CLICK]
Un arco viene definito incorreto se si verifica tale condizione [fai vedere sulla slide]
[CLICK]
Un vertice invece è incorretto per Max se ha almeno un arco uscente incorretto, per min se tutti i suoi archi uscenti sono incorretti.
-->

---
layout: center
---

<div class="border-white border-1 px-6">

$\textbf{function } \text{Init()} \\$
$\quad \textbf{for } u \in V \textbf{ do} \\$ 
$\quad \quad \mu(u) \gets 0 \\$

<v-click>

$\quad \textbf{for } u \in V_\text{Min} \textbf{ do} \\$ 
$\quad \quad \textbf{for } u \xrightarrow{w} v \in E \textbf{ do} \\$ 
$\quad \quad \quad \textbf{if }incorrect: \mu(u) < \delta(\mu(u), w) \textbf{ then} \\$ 
$\quad \quad \quad \quad Count(u) \gets Count(u) + 1 \\$
$\quad \quad \quad \textbf{if } Count(u) = Degree(u) \textbf{ then} \\$
$\quad \quad \quad \quad \text{Add } u \text{ to Incorrect} \\$ 

</v-click>
<v-click>

$\quad \textbf{for } u \in V_\text{Max} \textbf{ do} \\$ 
$\quad \quad \textbf{for } u \xrightarrow{w} v \in E \textbf{ do} \\$ 
$\quad \quad \quad \textbf{if }incorrect: \mu(u) < \delta(\mu(u), w) \textbf{ then} \\$ 
$\quad \quad \quad \quad \text{Add } u \text{ to Incorrect} \\$ 

</v-click>
</div>

<!--
L'algoritmo è formato da varie funzioni. La prima funzione è quella di inizializzazione, in cui vengono inizializzate le variabili, quindi la funzione di valore viene settata a 0 per ogni nodo, [CLICK] e poi viene fatta una prima passata su tutti i nodi di Min [CLICK] e di max per vedere quali di questi sono incorretti, considerando la condizione che abbiamo appena definito.
-->

---
layout: center
---

<div class="border-white border-1 mb-10 px-6">

$\textbf{function } \text{Treat(u)} \\$
$\quad \mu(u) \gets \mathbb{O}^\mathcal{G}(\mu)(u)$
</div>
<div class="border-white border-1 px-6">

$\textbf{function } \text{Update(u)} \\$
<v-click>

$\quad \textbf{if } u \in V_\text{Min} \textbf{ then} \\$
$\quad \quad Count(u) \gets 0\\$

</v-click>

<v-click>

$\quad \textbf{for } v \xrightarrow{w} u \in E \text{ which is incorrect} \textbf{ do} \\$ 
$\quad \quad \textbf{if } u \in V_\text{Min} \textbf{ then} \\$
$\quad \quad \quad Count(v) \gets Count(v) + 1\\$
$\quad \quad \quad \textbf{if } Count(v) = Degree(v) \textbf{ then} \\$
$\quad \quad \quad \quad \quad \text{Add } v \text{ to Incorrect} \\$

</v-click>
</div>

<!--
La seconda funzione si chiama "Treat", la quale prende in input un nodo $u$ e assegnare al valore di tale nodo il risultato della funzione O^g calcolata su $u$. 

La terza funzione è "Update", la quale prende in input un nodo $u$ [CLICK] [CLICK] e si occupa di aggiungere alla lista dei vertici incorretti i vertici incorretti appartenenti a Min che hanno un arco entrante in $u$.
-->

---
layout: center
---

<div class="border-white border-1 px-6">

$\textbf{function } \text{Main()} \\$
$\quad \text{Init()} \\$
$\quad \textbf{for } i = 0,1,2,\dots \textbf{ do} \\$
$\quad \quad Incorrect' \gets \emptyset \\$ 

<v-click>

$\quad \quad \textbf{for } u \in Incorrect \textbf{ do} \\$ 
$\quad \quad \quad \text{Treat(u)}  \\$
$\quad \quad \quad \text{Update(u)}  \\$

</v-click>

<v-click>

$\quad \quad \textbf{if } Incorrect' = \emptyset \textbf{ then} \\$ 
$\quad \quad \quad \textbf{return } \mu \\$
$\quad \quad \textbf{else } \\$
$\quad \quad \quad Incorrect \gets Incorrect' \\$

</v-click>
</div>

<!--
Infine abbiamo la funzione principale Main, la quale esegue l'inizializzazione [CLICK] e poi ripete un ciclo in cui ogni volta vengono esaminati solamente i nodi incorretti, aggiorando il valore e la nuova lista di nodi incorretti. Questo lo si fa sfruttando una lista temporanea "Incorrect'" che ogni volta viene azzerata ed il suo valore viene assegnato alla lista "Incorrect". [CLICK] Quando si arriva ad una iterazione in cui non si aggiunge nessun vertice alla lista temporanea, allora l'algoritmo termina e ritorna la funzione di valore.
-->

---

# Analisi della complessità
<v-clicks>

- La funzione `Init` ha una complessità di $O(n + |Out(V_\text{Min})| + |Out(V_\text{Max}|))$, dove $|Out(V)|$ è il numero totale di archi uscenti da $V$.

- La funzione `Treat(u)` equivale a calcolare $\mathbb{O}^\mathcal{G}(\mu)(u)$, quindi la complessità per ogni $u \in Incorrect$ è $O(n^2p) = O(m)$, considerando il caso peggiore in cui tutti i vertici sono incorretti (quindi $|Incorrect| = n$).

- La funzione `Update(u)` ha una complessità di $O(|In(u)|)$, che equivale a $O(np)$. Quindi applicata a tutti i vertici in `Incorrect` ha una complessità di $O(n^2p) = O(m)$

</v-clicks>

<v-click>

Possiamo vedere che nel caso peggiore in cui $|Incorrect| = n$ la complessità è $O(nmW)$, uguale all'approccio nativo.

</v-click>

<v-click>

Empiricamente però abbiamo che $|Incorrect| < n$, quindi l'algoritmo è più efficiente.

</v-click>

<!--
Per analizzare la complessità computazionale dell'intero algoritmo iniziamo analizzando la complessità delle funzioni che lo compongono.
[CLICK]
La funzione Init ha una complessità di O(n) sommata al numero di archi uscenti dai nodi di min e di max.
[CLICK]
La funzione Treat equivale a calcolare O^g e quindi la sua complessità per un singolo nodo è O(m).
[CLICK]
La funzione Update ha una complessità pari al numero di archi entranti da $u$, la quale equivale a O(np).
[CLICK]
Nel caso in cui Incorrect include sempre tutti i vertici, allora la complessità è uguale a O(nmW). 
[CLICK] 
Empiricamente però abbiamo che il numero di nodi in Incorrect per ogni iterazione è sempre minore di $n$, e quindi l'algoritmo è più efficiente.
-->

---

# Implementazione Python

<div class="no-scrollbar">

```python{1-4|6-29|30-31|32-48|48-66}{maxHeight:'500px'} 
def optimized_value_iteration(self):
    incorrect: Set[int] = set() 
    incorrect_prime: Set[int] = set()
    count: Dict[int, int] = {node: 0 for node in self.arena.nodes}

    def init():
        self.arena.value_mapping = {node: 0 for node in self.arena.nodes}
        pbar = tqdm(total=len(self.arena.nodes), desc="Opt Value Iteration - Init")
        # For each MIN node
        min_nodes = (n for n in self.arena.nodes 
                        if self.arena.player_mapping[n] == Player.MIN)
        for node in min_nodes:
            pbar.update(1)
            for (u, v, w) in self.arena.get_outgoing_edges(node):
                if self.arena.value_mapping[u] < self._delta(self.arena.value_mapping[v], w):
                    count[u] += 1
            # If count == degree of node
            if count[node] == self.arena.get_node_degree(node):
                incorrect.add(node)

        # For each MAX node
        max_nodes = (n for n in self.arena.nodes
                        if self.arena.player_mapping[n] == Player.MAX)
        for node in max_nodes:
            pbar.update(1)
            for (u, v, w) in self.arena.get_outgoing_edges(node):
                if self.arena.value_mapping[u] < self._delta(self.arena.value_mapping[v], w):
                    incorrect.add(u)
    
    def treat(u: int):
        self.arena.value_mapping[u] = self._O(u)

    def update(u: int):
        if self.arena.player_mapping[u] == Player.MIN:
            count[u] = 0

        for (v, _, w) in self.arena.ingoing_edges.get(u, set()):
            # Only consider nodes that are still incorrect
            if not (self.arena.value_mapping[v] < self._delta(self.arena.value_mapping[u], w)):
                continue

            if self.arena.player_mapping[v] == Player.MIN:
                count[v] += 1
                if count[v] == self.arena.get_node_degree(v):
                    incorrect_prime.add(v)
            if self.arena.player_mapping[v] == Player.MAX:
                incorrect_prime.add(v)

    init()
    n = len(self.arena.nodes)
    W = self.arena.max_weight
    max_steps = n * W
    steps = 0
    for i in tqdm(range(max_steps)):
        steps += 1
        incorrect_prime = set()

        for u in incorrect:
            treat(u)
            update(u)

        if incorrect_prime == set():
            print(f"Converged after {i} steps")
            return steps
        incorrect = incorrect_prime 
    return steps
```
</div>

<!-- Come prima, qui abbiamo l'implementazione dell'algoritmo in Python. Vengono quindi definite le strutture dati essenziali, e poi le singole funzioni. --> 

---
layout: section
---

# Generazione dell'arena
Come generare un'arena valida per il gioco

<!--
Abbiamo visto come, data un'arena di gioco, è possibile calcolare la funzione di valore. Ora vediamo in che maniera è possibile generare un'arena valida per il gioco.

Come abbiamo accennato, il problema principale è il fatto che il grafo non deve contenere cicli negativi.
-->

---

# Generazione dell'arena

### Parametri
- `num_nodes`: numero di nodi;
- `edge_probability`: probabilità di avere un arco tra due nodi;
- `max_weight`: definisce il range $[-W, W]$ dei pesi degli archi;

<!--
Per generare l'arena abbiamo bisogno dei seguenti parametri, quindi numero di nodi, probabiltà di avere un arco tra due nodi e il massimo peso di un arco.
-->

---
layout: center
---

<div class='border-1 px-6'>

$edges \gets \emptyset \\$
$\textbf{for } v \in V \textbf{ do} \\$
$\quad \textbf{for } u \in V \textbf{ do} \\$
$\quad \text{random\_number} \gets random(0, 1) \\$
$\quad \quad \textbf{if } \text{random\_number} \le \text{edge\_probability} \textbf{ do}\\$

<v-clicks>

$\quad \quad \quad w \gets sample(-W, W) \\$
$\quad \quad \quad edge \gets (v, u, w) \\$

</v-clicks>
<v-clicks>

$\quad \quad \quad \textbf{if } edge \text{ doesn't create a cycle} \textbf{ do}\\$
$\quad \quad \quad \quad edges \gets edges \cup \{(v, u,w)\}$

</v-clicks>
</div>

<!--
Questo è l'algoritmo ad alto livello per la generazione dell'arena, in cui esaminiamo tutte le coppie di nodi, estraiamo un numero casuale tra 0 ed 1 e se questo numero è minore della edge_probability, allora [CLICK] campioniamo un peso dall'intervallo, creiamo un arco con quel peso [CLICK] ed aggiungiamo l'arco all'insieme di archi solo se questo non crea un ciclo negativo.
-->

---

# Bellman Ford
- Capire se il grafo presenta cicli negativi.

<v-clicks>

- Ogni volta che viene eseguito, deve visitare tutti i nodi e tutti gli archi.
- Complessità di $O(mn)$.

</v-clicks>

<v-click>

```python {1-5|6-10|11-16}{maxHeight:'500px'}
def bellman_ford(self):
    """
    Detect negative cycles using Bellman - Ford algorithm .
    """
    distances = { node:0 for node in nodes }
    # Relax edges repeatedly
    for _ in range (len(nodes)-1):
        for edge in edges:
            if distances [edge[0]] + edge[2] < distances.get(edge[1],float('inf')):
                distances[edge[1]] = distances[edge[0]] + edge[2]
    # Check for negative cycles
    for edge in edges:
        if distances[edge[0]] + edge[2] < distances.get(edge[1],float('inf')):
    # Negative cycle found
            return True 
    return False
```

</v-click>

<!--
Il primo approccio per capire se il grafo presenta cicli negativi è quello di usare l'algoritmo di Bellman Ford.
[CLICK]
Il problema di questo algoritmo è che ogni volta che viene eseguito, deve visitare tutti i nodi e tutti glio archi [CLICK] ed ha quindi una complessità di O(mn). Visto che durante la generazione dell'arena andrebbe eseguito una volta per ogni arco, si otterrebbe una complessità molto alta.
[CLICK]
Questa è l'impmenetazione dell'algoritmo di Bellman Ford in Python. L'algoritmo utilizza un dizionario delle distanze che viene inizializzato ogni volta che la funzione viene chiamata, [CLICK] vengono poi rilassati tutti gli archi $n$ volte [CLICK] ed infine si itera su tutti gli archi per trovare un ciclo negativo.
-->

---

# Bellman Ford Incrementale
 
- Ad ogni step, un solo arco nuovo.

<v-clicks>

- Aggiornare solamente i pesi relativi al sottografo creato dall'arco.

</v-clicks>

<v-click>

```python {0-15|16-19|20-35}{maxHeight:'300px'}
def bellman_ford_incremental(self, new_edge: Tuple[int, int, float]) -> bool:
    """
    A very efficient implementation of the Bellman-Ford algorithm 
    that only checks for negative cycles related to the new edge. 
    It uses the fast_edges dictionary to keep track of the edges 
    and their weights, in order to avoid iterating over all the edges.
    """
    # Add the new edge
    self.edges.add(new_edge)
    self.fast_edges[new_edge[0]][new_edge[1]] = new_edge[2]

    new_distance_0 = self.distances.get(new_edge[0], 0) + new_edge[2]
    new_distance_1 = self.distances.get(new_edge[1], float('inf'))

    previous_distance_1 = self.distances.get(new_edge[1], None)

    # Relax edges related to the new edge
    if new_distance_0 < new_distance_1:
        new_distance_1 = new_distance_0

    self.distances[new_edge[1]] = new_distance_1

    origin_to = self.fast_edges[new_edge[0]] #this is a dict that maps the destination to the weight
    dest_to = self.fast_edges[new_edge[1]] #this is a dict that maps the origin to the weight

    edges = {(new_edge[0], dest, weight) for dest, weight in origin_to.items()} | \
        {(new_edge[1], origin, weight) for origin, weight in dest_to.items()}
    for edge in edges:
        if self.distances[edge[0]] + edge[2] < self.distances.get(edge[1], float('inf')):
            self.edges.remove(new_edge)
            self.fast_edges[new_edge[0]].pop(new_edge[1])
            self.distances[new_edge[1]] = previous_distance_1
            return True

    return False  # No negative cycle found
```

</v-click>

<v-click>

Complessità di $O(np)$.

</v-click>
<!--
Analizzando l'algoritmo, l'idea per migliorare le performances è quella di 
memorizzare i valori calcolati negli step precedenti, [CLICK] ed ogni volta che un nuovo arco deve essere aggiunto consideriamo solo il sottoinsieme di archi collegati a tale arco per vedere se questo ha creato un nuovo ciclo negativo. 

[CLICK]
In questo nuovo algoritmo quindi si mantiene in memoria il dizionario delle distanze che viene aggiornato di volta in volta. [CLICK] Si rilassa solamente il nuovo arco [CLICK] ed infine si itera solamente sugli archi entranti all'origine ed uscenti alla destinazione del nuovo arco per vedere se il nuovo arco crea un ciclo negativo.

[CLICK]
In questa maniera passiamo da una complessità computazionel di O(mn) per chiamata ad una complessità O(np), ovvero il numero massimo di nodi entranti o uscenti da un determinato nodo.

-->

---
layout: section
---
# Performance Evaluation
Valutare i tempi di generazione dell'arena e risoluzione del gioco con i differenti algoritmi

<!-- Per valutare le performance degli algoritmi di cui ho parlato ho effettuato varie prove con diverse configurazioni, che comprendono il numero di nodi e la probabilità di avere un arco tra due nodi. Per ogni configurazione ho eseguito i test circa cinque volte in modo da fare una media. -->

---
layout: center
---
<div class="pt-20">

# Generazione dell'arena

</div>

<div class="h-[500px] px-10 pb-20 overflow-y-scroll">

| Number of nodes | Edge probability | Time (no checks) | Time (BF)  | Time (Optimized BF) |
|-----------------|------------------|------------------|------------|---------------------|
| 100             | 0.1              | 8.80 ms          | 27.43 ms   | 15.13 ms            |
| 100             | 0.2              | 7.39 ms          | 42.31 ms   | 14.38 ms            |
| 100             | 0.3              | 9.29 ms          | 57.22 ms   | 19.54 ms            |
| 100             | 0.4              | 8.31 ms          | 72.40 ms   | 25.35 ms            |
| 100             | 0.5              | 11.57 ms         | 88.39 ms   | 34.94 ms            |
| 200             | 0.1              | 10.53 ms         | 133.68 ms  | 20.90 ms            |
| 200             | 0.2              | 12.71 ms         | 252.16 ms  | 44.88 ms            |
| 200             | 0.3              | 15.09 ms         | 365.00 ms  | 70.19 ms            |
| 200             | 0.4              | 17.11 ms         | 490.65 ms  | 108.65 ms           |
| 200             | 0.5              | 19.67 ms         | 603.05 ms  | 158.93 ms           |
| 300             | 0.1              | 18.54 ms         | 413.68 ms  | 46.43 ms            |
| 300             | 0.2              | 22.73 ms         | 824.86 ms  | 113.85 ms           |
| 300             | 0.3              | 30.61 ms         | 1.21 sec   | 202.74 ms           |
| 300             | 0.4              | 32.01 ms         | 1.62 sec   | 333.74 ms           |
| 300             | 0.5              | 45.70 ms         | 2.00 sec   | 489.24 ms           |
| 400             | 0.1              | 28.15 ms         | 1.03 sec   | 92.53 ms            |
| 400             | 0.2              | 36.78 ms         | 1.98 sec   | 236.87 ms           |
| 400             | 0.3              | 55.51 ms         | 2.90 sec   | 446.53 ms           |
| 400             | 0.4              | 59.05 ms         | 3.98 sec   | 749.61 ms           |
| 400             | 0.5              | 68.06 ms         | 4.82 sec   | 1.13 sec            |
| 500             | 0.1              | 41.80 ms         | 1.95 sec   | 154.65 ms           |
| 500             | 0.2              | 62.84 ms         | 3.84 sec   | 463.09 ms           |
| 500             | 0.3              | 76.73 ms         | 5.58 sec   | 847.26 ms           |
| 500             | 0.4              | 87.64 ms         | 7.49 sec   | 1.39 sec            |
| 500             | 0.5              | 98.93 ms         | 9.38 sec   | 2.18 sec            |
| 1000            | 0.1              | 161.31 ms        | 15.41 sec  | 913.22 ms           |
| 1000            | 0.2              | 214.72 ms        | 29.93 sec  | 3.06 sec            |
| 1000            | 0.3              | 281.77 ms        | 44.60 sec  | 5.88 sec            |
| 1000            | 0.4              | 333.74 ms        | 59.64 sec  | 10.57 sec           |
| 1000            | 0.5              | 392.45 ms        | 1.24 min   | 18.48 sec           |
| 2000            | 0.1              | 616.58 ms        | 2.11 min   | 6.18 sec            |
| 2000            | 0.2              | 871.34 ms        | 4.01 min   | 22.53 sec           |
| 2000            | 0.3              | 1.10 sec         | 6.00 min   | 45.92 sec           |
| 2000            | 0.4              | 1.32 sec         | 8.03 min   | 1.36 min            |
| 2000            | 0.5              | 1.57 sec         | 9.93 min   | 2.14 min            |
| 5000            | 0.1              | 3.98 sec         | 32.20 min  | 1.51 min            |
| 5000            | 0.2              | 5.49 sec         | 62.57 min  | 5.44 min            |
| 5000            | 0.3              | 7.33 sec         | 94.75 min  | 11.73 min           |
| 5000            | 0.4              | 8.90 sec         | 125.82 min | 20.78 min           |
| 5000            | 0.5              | 11.14 sec        | 155.92 min | 33.99 min           |

</div>

<!-- In questa tabella troviamo i tempi di generazione dell'arena. Nelle prime due colonne ci sono numero di nodi e probabilità di avere un arco, mentre nelle ultime tre abbiamo il tempo di generazione senza alcun controllo sui cicli negativi; poi il tempo di generazione utilizzando Bellman Ford classico ed infine Bellman Ford ottimizzato. 

Possiamo vedere come il controllo sui cicli negativi incrementi di molto il temop di generazione dell'arena, ma anche come l'algoritmo di Bellman Ford ottimizzato impieghi molto meno tempo rispetto alla versione originale. -->
---

# Risoluzione del gioco

<div class="h-[500px] px-10 pb-20 overflow-y-scroll">

| Number of nodes | Edge probability | Time (naive) | Time (optimized) |
| --------------- | ---------------- | ------------ | ---------------- |
| 10              | 0.1              | 2.80 ms      | 2.66 ms          |
| 10              | 0.2              | 3.63 ms      | 2.80 ms          |
| 10              | 0.5              | 2.34 ms      | 3.44 ms          |
| 50              | 0.1              | 3.05 ms      | 2.89 ms          |
| 50              | 0.2              | 3.41 ms      | 3.19 ms          |
| 50              | 0.5              | 4.86 ms      | 3.68 ms          |
| 100             | 0.1              | 3.65 ms      | 3.28 ms          |
| 100             | 0.2              | 4.49 ms      | 4.53 ms          |
| 100             | 0.3              | 6.68 ms      | 4.83 ms          |
| 100             | 0.4              | 7.15 ms      | 7.75 ms          |
| 100             | 0.5              | 8.58 ms      | 7.09 ms          |
| 200             | 0.1              | 7.42 ms      | 5.46 ms          |
| 200             | 0.2              | 14.09 ms     | 10.01 ms         |
| 200             | 0.3              | 21.12 ms     | 16.05 ms         |
| 200             | 0.4              | 32.17 ms     | 26.29 ms         |
| 200             | 0.5              | 39.86 ms     | 32.95 ms         |
| 300             | 0.1              | 17.05 ms     | 15.46 ms         |
| 300             | 0.2              | 31.63 ms     | 35.43 ms         |
| 300             | 0.3              | 58.42 ms     | 45.91 ms         |
| 300             | 0.4              | 99.49 ms     | 82.58 ms         |
| 300             | 0.5              | 161.07 ms    | 113.33 ms        |
| 400             | 0.1              | 33.19 ms     | 26.94 ms         |
| 400             | 0.2              | 81.50 ms     | 64.60 ms         |
| 400             | 0.3              | 151.79 ms    | 110.84 ms        |
| 400             | 0.4              | 230.66 ms    | 172.51 ms        |
| 400             | 0.5              | 361.59 ms    | 285.20 ms        |
| 500             | 0.1              | 58.26 ms     | 43.85 ms         |
| 500             | 0.2              | 142.15 ms    | 111.85 ms        |
| 500             | 0.3              | 365.51 ms    | 291.57 ms        |
| 500             | 0.4              | 513.59 ms    | 429.98 ms        |
| 500             | 0.5              | 638.67 ms    | 505.26 ms        |
| 500             | 1.0              | 1.85 sec     | 1.37 sec         |
| 1000            | 0.1              | 309.54 ms    | 225.22 ms        |
| 1000            | 0.2              | 1.03 sec     | 886.58 ms        |
| 1000            | 0.3              | 1.94 sec     | 1.47 sec         |
| 1000            | 0.4              | 3.79 sec     | 2.63 sec         |
| 1000            | 0.5              | 4.95 sec     | 4.21 sec         |
| 2000            | 0.1              | 2.35 sec     | 1.73 sec         |
| 2000            | 0.2              | 8.46 sec     | 7.04 sec         |
| 2000            | 0.3              | 15.62 sec    | 14.01 sec        |
| 2000            | 0.4              | 26.91 sec    | 23.65 sec        |
| 2000            | 0.5              | 37.48 sec    | 32.75 sec        |
| 5000            | 0.1              | 39.75 sec    | 33.81 sec        |
| 5000            | 0.2              | 2.24 min     | 2.15 min         |
| 5000            | 0.3              | 4.48 min     | 4.42 min         |
| 5000            | 0.4              | 8.72 min     | 7.65 min         |
| 5000            | 0.5              | 15.11 min    | 13.17 min        |

</div>

<!-- In quest'altra tabella invece abbiamo i risultati relativi alla risoluzione del gioco. Come prima sono mostrati numero di nodi e probabilità degli archi, e sono messi a confronto i tempi tra l'algoritmo naive e quello ottimizzato. Visto che la complessità asintotica è la stessa la differenza non è enorme, però possiamo vedere come in generale l'algoritmo ottimizzato è più veloce rispetto a quello naive. -->

---
layout: center
---

# Conclusioni
- Risolutore di energy games
- Generazione arena senza cicli negativi

<!-- Per concludere quindi abbiamo visto che è possibile generare un'arena di gioco valida per gli energy games sfruttando algoritmi come Bellman Ford, e trovare la funzione di valore con un algoritmo naive ed uno più ottimizzato in tempi ragionevoli soprattutto per grafi non troppo densi. -->

---
layout: center
---

# Grazie per l'attenzione

<!-- Questo è tutto, grazie per l'attenzione. -->
